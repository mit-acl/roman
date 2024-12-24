###########################################################
#
# mapper_params.py
#
# Params for ROMAN open-set segment mapping
#
# Authors: Mason Peterson
#
# Dec. 21, 2024
#
###########################################################

from dataclasses import dataclass
import yaml
from typing import Tuple

from robotdatapy.camera import CameraParams

@dataclass
class MapperParams():
    """
    Params for ROMAN open-set segment mapper object.
    
    Args:
        min_iou (float): minimum voxel IOU for frame-to-frame association
        min_sightings (int): minimum number of sightings to consider an object
        max_t_no_sightings (int): maximum time without a sighting before moving 
            an object to inactive
        merge_objects_iou_3d (float): minimum 3D IOU for merging objects
        merge_objects_iou_2d (float): minimum 2D IOU for merging objects
        mask_downsample_factor (int): factor by which to downsample the mask
        min_max_extent (float): to get rid of very small objects
        plane_prune_params (Tuple[float]): to get rid of planar objects (likely background)
        segment_graveyard_time (float): time after which an inactive segment is sent to the graveyard
        segment_graveyard_dist (float): distance traveled after which an inactive segment is sent to the graveyard
        iou_voxel_size (float): voxel size for IOU calculation
        segment_voxel_size (float): voxel size for segment representation

    Returns:
        MapperParams: params object
    """

    min_iou: float = 0.25
    min_sightings: int = 2
    max_t_no_sightings: int = 0.4
    merge_objects_iou_3d: float = 0.25
    merge_objects_iou_2d: float = 0.8
    mask_downsample_factor: int = 8
    min_max_extent: float = 0.25
    plane_prune_params: Tuple[float] = (3.0, 3.0, 0.5)
    segment_graveyard_time: float = 15.0
    segment_graveyard_dist: float = 10.0
    iou_voxel_size: float = 0.2
    segment_voxel_size: float = 0.05
    
    @classmethod
    def from_yaml(cls, yaml_path: str, run: str = None):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if run is not None:
            data = data[run]
        return cls(**data)