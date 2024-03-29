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
    pixel: np.ndarray
    width: int
    height: int
    mask: np.ndarray = None
    keypoints: List[cv.KeyPoint] = None
    descriptors: np.ndarray = None

    def copy(self, include_mask: bool = True):
        if include_mask:
            return Observation(self.time, self.pose.copy(), self.pixel.copy(), 
                               self.width, self.height, self.mask.copy())
        else:
            return Observation(self.time, self.pose.copy(), self.pixel.copy(), 
                           self.width, self.height, None)