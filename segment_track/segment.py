import numpy as np
import gtsam
from typing import List

from robot_utils.robot_data.img_data import CameraParams
from robot_utils.transform import transform
from robot_utils.camera import xyz_2_pixel

import open3d as o3d
from segment_track.observation import Observation

class Segment():

    def __init__(self, observation: Observation, camera_params: CameraParams, 
                 pixel_std_dev: float, id: int = 0, voxel_size: float = 0.05):
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
        self.points = None
        self.voxel_size = voxel_size  # voxel size used for maintaining point clouds

    def update(self, observation: Observation, force=False, integrate_points=True):
        """Update a 3D segment with a new observation

        Args:
            observation (Observation): Input observation object
            force (bool, optional): If true, do not attempt 3D reconstruction. Defaults to False.
            integrate_points (bool, optional): If true, integrate point cloud contained in observation. Defaults to True.
        """

        # See if this will break the reconstruction
        if not force:
            try: 
                self.reconstruction_from_observations(self.observations + [observation], width_height=False)
            except:
                return
            
        # Integrate point measurements
        if integrate_points:
            self.integrate_points_from_observation(observation)

        self.num_sightings += 1
        if observation.time > self.last_seen:
            self.observations.append(observation.copy(include_mask=True))
            self.last_seen = observation.time
            self.last_observation.mask = None
            self.last_observation = observation
        else:
            self.observations.append(observation.copy(include_mask=False))
    
    def integrate_points_from_observation(self, observation: Observation):
        """Integrate point cloud in the input observation object

        Args:
            observation (Observation): input observation object
        """
        if observation.point_cloud is None:
            return
        # Convert observed points to global frame
        Twb = observation.pose
        Rwb = Twb[:3,:3]
        twb = Twb[:3,3].reshape((3,1))
        points_obs_body = observation.point_cloud.T
        num_points_obs = points_obs_body.shape[1]
        points_obs_world = Rwb @ points_obs_body + np.repeat(twb, num_points_obs, axis=1)
        self._add_points(points_obs_world.T)

    def integrate_points_from_segment(self, segment):
        """Integrate the points from another segment into this segment.
        Both segments are assumed to be in the same reference frame

        Args:
            segment (Segment): _description_
        """
        if segment.num_points > 0:
            self._add_points(segment.points)

    def _add_points(self, points):
        assert points.shape[1] == 3
        if points.shape[0] == 0:
            return
        if self.points is None:
            self.points = points
        else:
            self.points = np.concatenate((self.points, points), axis=0)
        self._cleanup_points()
    
    def _cleanup_points(self):
        if self.points is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points.extend(self.points)
            pcd_sampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            pcd_pruned, _ = pcd_sampled.remove_statistical_outlier(10, 0.1)
            if pcd_pruned.is_empty():
                self.points = None
            else:
                self.points = np.asarray(pcd_pruned.points)    

    @property
    def num_points(self):
        if self.points is None:
            return 0
        else:
            return self.points.shape[0]
        
    def aabb_volume(self):
        """Return the volume of the 3D axis-aligned bounding box
        """
        if self.num_points > 0:
            aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(self.points)
            )
            return aabb.volume()
        return 0.0

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
