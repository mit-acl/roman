import numpy as np
import gtsam

from robot_utils.robot_data.img_data import CameraParams
from robot_utils.transform import transform

from segment_track.observation import Observation

class Segment():

    def __init__(self, observation: Observation, camera_params: CameraParams, 
                 pixel_std_dev: float, id: int = 0):
        self.id = id
        if observation.mask is not None:
            self.last_mask = observation.mask.copy()
        else:
            self.last_mask = None
        self.observations = [observation.copy(include_mask=False)]
        self.last_seen = observation.time
        self.camera_params = camera_params
        self.cal3ds2 = gtsam.Cal3DS2(camera_params.K[0,0], camera_params.K[1,1], 
            camera_params.K[0,1], camera_params.K[0,2], camera_params.K[1,2], 
            camera_params.D[0], camera_params.D[1], camera_params.D[2], camera_params.D[3])
        self.pixel_std_dev = pixel_std_dev
        self.num_sightings = 1

    def update(self, observation: Observation):
        if observation.mask is not None:
            self.last_mask = observation.mask.copy()
        self.observations.append(observation.copy(include_mask=False))
        self.last_seen = observation.time
        self.num_sightings += 1

    def reconstruction3D(self, width_height=False):
        camera_poses = []
        for obs in self.observations:
            camera_poses.append(gtsam.PinholeCameraCal3DS2(gtsam.Pose3(obs.pose), self.cal3ds2))
        camera_set = gtsam.gtsam.CameraSetCal3DS2(camera_poses)
        pixels_point_vec = gtsam.Point2Vector([gtsam.Point2(obs.pixel) for obs in self.observations])
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.pixel_std_dev)
        reconstruction = gtsam.triangulatePoint3(camera_set, pixels_point_vec, rank_tol=1e-9, optimize=True, model=measurement_noise)

        if width_height:
            ws, hs = [], []
            for obs in self.observations:
                p_c = transform(np.linalg.inv(obs.pose), reconstruction)
                ws.append(obs.width * np.abs(p_c[0]) / np.abs(obs.pixel[0] - self.camera_params.K[0,2]))
                hs.append(obs.height * np.abs(p_c[1]) / np.abs(obs.pixel[1] - self.camera_params.K[1,2]))
            return np.concatenate((reconstruction, [np.mean(ws)], [np.mean(hs)]))
                
        return reconstruction