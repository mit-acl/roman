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
    mask_downsampled: np.ndarray = None
    point_cloud: np.ndarray = None  # n-by-3 matrix. Each row is a 3D point.

    def copy(self, include_mask: bool = True, include_ptcld = False):
        ptcld_copy = None
        if self.point_cloud is not None and include_ptcld:
            ptcld_copy = self.point_cloud.copy()
        if include_mask:
            return Observation(self.time, self.pose.copy(), self.pixel.copy(), 
                               self.width, self.height, self.mask, self.mask_downsampled, ptcld_copy)
        else:
            return Observation(self.time, self.pose.copy(), self.pixel.copy(), 
                           self.width, self.height, None, None, ptcld_copy)