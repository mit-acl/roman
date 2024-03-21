from dataclasses import dataclass
import numpy as np

@dataclass
class Observation():
    """
    Segment observation data class
    """

    time: float
    pose: np.ndarray
    pixel: np.ndarray
    width: int
    height: int
    mask: np.ndarray = None
    point_cloud: np.ndarray = None

    def copy(self, include_mask: bool = True):
        if include_mask:
            return Observation(self.time, self.pose.copy(), self.pixel.copy(), 
                               self.width, self.height, self.mask.copy(), self.point_cloud.copy())
        else:
            return Observation(self.time, self.pose.copy(), self.pixel.copy(), 
                           self.width, self.height, None, self.point_cloud.copy())