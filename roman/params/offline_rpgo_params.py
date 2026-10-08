###########################################################
#
# offline_rpgo_params.py
#
# Params for ROMAN offline RPGO SLAM.
#
# Authors: Mason Peterson
#
# Jan. 15, 2025
#
###########################################################

import numpy as np

from dataclasses import dataclass
from typing import List
import os
import yaml

@dataclass
class OfflineRPGOParams:
    
    # odometrys closure covariance params
    odom_t_std: float = 0.1
    odom_r_std: float = np.deg2rad(0.5)
    
    # loop closure covariance params
    lc_t_std: float = 1.0
    lc_r_std: float = np.deg2rad(2.0)
    
    # sparse or dense
    sparsified: bool = True

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            params = yaml.full_load(f)
        return cls(**params)