import numpy as np
import gtsam
from typing import List

from robot_utils.robot_data.img_data import CameraParams
from robot_utils.transform import transform
from robot_utils.camera import xyz_2_pixel

from segment_track.observation import Observation

class Segment():

    def __init__(self, observation: Observation, camera_params: CameraParams, 
                 pixel_std_dev: float, id: int = 0):
        self.id = id
        self.observations = [observation.copy(include_mask=False)]
        self.last_seen = observation.time
        self.camera_params = camera_params
        self.cal3ds2 = gtsam.Cal3DS2(camera_params.K[0,0], camera_params.K[1,1], 
            camera_params.K[0,1], camera_params.K[0,2], camera_params.K[1,2], 
            camera_params.D[0], camera_params.D[1], camera_params.D[2], camera_params.D[3])
        self.pixel_std_dev = pixel_std_dev
        self.num_sightings = 1
        self.edited = True
        self.last_observation = observation

    def update(self, observation: Observation, force=False):

        # See if this will break the reconstruction
        if not force:
            try: 
                self.reconstruction_from_observations(self.observations + [observation], width_height=False)
            except:
                return

        self.num_sightings += 1
        if observation.time > self.last_seen:
            self.observations.append(observation.copy(include_mask=False))
            self.last_seen = observation.time
            self.last_observation = observation.copy(include_mask=True)
        else:
            self.observations.append(observation.copy(include_mask=False))

    @property
    def last_mask(self):
        if self.last_observation.mask is not None:
            return self.last_observation.mask.copy()
        else:
            return None

    def reconstruction3D(self, width_height=False):

        if not self.edited and (not width_height or len(self.reconstruction) > 3):
            if not width_height:
                return self.reconstruction[:3]
            return self.reconstruction
        
        self.reconstruction = self.reconstruction_from_observations(self.observations, width_height)
        return self.reconstruction
    
    def reconstruction_from_observations(self, observations: List[Observation], width_height=False):
        camera_poses = []
        for obs in observations:
            camera_poses.append(gtsam.PinholeCameraCal3DS2(gtsam.Pose3(obs.pose), self.cal3ds2))
        camera_set = gtsam.gtsam.CameraSetCal3DS2(camera_poses)
        pixels_point_vec = gtsam.Point2Vector([gtsam.Point2(obs.pixel) for obs in observations])
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.pixel_std_dev)
        reconstruction = gtsam.triangulatePoint3(camera_set, pixels_point_vec, rank_tol=1e-9, optimize=True, model=measurement_noise)

        if width_height:
            ws, hs = [], []
            for obs in observations:
                p_c = transform(np.linalg.inv(obs.pose), reconstruction)
                ws.append(obs.width * np.abs(p_c[2]) / self.camera_params.K[0,0])
                hs.append(obs.height * np.abs(p_c[2]) / self.camera_params.K[1,1])
            reconstruction = np.concatenate((reconstruction, [np.mean(ws)], [np.mean(hs)]))
                
        self.reconstruction = reconstruction
        return reconstruction
    
    def reconstruct_mask(self, pose):
        mask = np.zeros((self.camera_params.height, self.camera_params.width), dtype=np.uint8)
        reconstruction = self.reconstruction3D(width_height=True)
        p_c = transform(np.linalg.inv(pose), reconstruction[:3])
        if p_c[2] < 0:
            return mask
        
        w, h = reconstruction[3], reconstruction[4]
        lower_left_c = p_c - np.array([w/2, h/2, 0])
        upper_right_c = p_c + np.array([w/2, h/2, 0])
        lower_left_uv = xyz_2_pixel(lower_left_c, self.camera_params.K)
        upper_right_uv = xyz_2_pixel(upper_right_c, self.camera_params.K)

        lower_left_uv = np.round(lower_left_uv).astype(int).reshape(-1)
        upper_right_uv = np.round(upper_right_uv).astype(int).reshape(-1)

        lower_left_uv = np.maximum(lower_left_uv, 0)
        upper_right_uv = np.minimum(upper_right_uv, [self.camera_params.width, self.camera_params.height])
        mask[lower_left_uv[1]:upper_right_uv[1], lower_left_uv[0]:upper_right_uv[0]] = 1

        return mask
    
    @property
    def covariance(self):
        graph = gtsam.NonlinearFactorGraph()
        initial_guess = gtsam.Values()
        segment = gtsam.symbol_shorthand.S(0)
        x = gtsam.symbol_shorthand.X

        for i, obs in enumerate(self.observations):

            pose_factor = gtsam.NonlinearEqualityPose3(x(i), gtsam.Pose3(obs.pose))
            graph.push_back(pose_factor)
            initial_guess.insert(x(i), gtsam.Pose3(obs.pose))

            measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.pixel_std_dev)
            projection_factor = gtsam.GenericProjectionFactorCal3DS2(
                gtsam.Point2(obs.pixel), measurement_noise, x(i), segment, self.cal3ds2)
            graph.push_back(projection_factor)
        initial_guess.insert(segment, gtsam.Point3(self.reconstruction3D(width_height=False)))

        optimizer = gtsam.GaussNewtonOptimizer(graph, initial_guess)

        result = optimizer.optimize()
        marginal_covariance = gtsam.Marginals(graph, result)

        return marginal_covariance.marginalCovariance(segment)
