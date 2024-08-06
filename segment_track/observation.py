from dataclasses import dataclass
import numpy as np
import cv2 as cv
from typing import List

@dataclass
class Observation():
    """
    Segment observation data class
    """

    time: float
    pose: np.ndarray
    mask: np.ndarray = None
    mask_downsampled: np.ndarray = None
    point_cloud: np.ndarray = None  # n-by-3 matrix. Each row is a 3D point.
    keypoints: List[cv.KeyPoint] = None
    descriptors: np.ndarray = None
    clip_embedding: np.ndarray = None

    def copy(self, include_mask: bool = True, include_ptcld = False):
        ptcld_copy = None
        if self.point_cloud is not None and include_ptcld:
            ptcld_copy = self.point_cloud.copy()
        if include_mask:
            return Observation(self.time, self.pose.copy(), 
                               self.mask, self.mask_downsampled, ptcld_copy)
        else:
            return Observation(self.time, self.pose.copy(), None, None, ptcld_copy)