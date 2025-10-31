###########################################################
#
# mapper_params.py
#
# Params for ROMAN open-set segment mapping
#
# Authors: Mason Peterson, Qingyuan Li
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
from dataclasses import dataclass
import yaml
from typing import Tuple, Union

@dataclass
class MapperParams():
    """
    Params for ROMAN open-set segment mapper object.
    
    Args:
        geometric_association_method (str): geometric method for associating segments across frames.
            ('chamfer', 'iou, 'iom') for chamfer distance, voxel IOU, or voxel Intersection-over-Minimum (IOM).
        semantic_association_method (str): semantic method for associating segments across frames.
            ('cosine_similarity', 'none') for cosine similarity or no semantic association.
        geometric_score_range (Tuple[float]): threshold and maximum geometric association score.
            The threshold is used for both frame-to-frame association and merging objects, while the maximum
            is used to normalize the association score using the formula ((score - threshold) / (max - threshold)) 
            to potentially combine with a normalized semantic score for frame-to-frame association. 
            The max should be 1 for IOU/IOM. Note that chamfer distance is negated, since smaller distances 
            are more similar, so the range should be negative, e.g. [-0.1, 0.0] for a threshold of 0.1m
            and 'maximum' similarity at 0 distance. 
        semantic_score_range (Tuple[float]): threshold and maximum semantic association score.
            Used the same way as geometric_score_range. The max should be -1 for cosine similarity.
            Only used if semantic_association_method is not None.
        min_2d_iou (float|None): minimum 2D IOU for merging objects. 2D IOU will not be used if set to None.
        min_sightings (int): minimum number of sightings to consider an object
        max_t_no_sightings (int): maximum time without a sighting before moving 
            an object to inactive
        mask_downsample_factor (int): factor by which to downsample the mask
        min_max_extent (float): to get rid of very small objects
        plane_prune_params (Tuple[float]): to get rid of planar objects (likely background)
        segment_graveyard_time (float): time after which an inactive segment is sent to the graveyard
        segment_graveyard_dist (float): distance traveled after which an inactive segment is sent to the graveyard
        iou_voxel_size (float): voxel size for IOU calculation (used if geometric_association_method is 'iou' or 'iom')
        segment_voxel_size (float): voxel size for segment representation (point-cloud downsampling)
        segment_outlier_removal_std (float): standard deviation for statistical outlier removal of 
            segment point clouds. If points are further than this many standard deviations from the mean
            distance to neighbors, they are considered outliers and removed. If <= 0 or inf, no outlier 
            removal is performed.

    Returns:
        MapperParams: params object
    """

    geometric_association_method: str = 'iou'
    semantic_association_method: str = 'none'
    geometric_score_range: Tuple[float] = (0.25, 1.0)
    semantic_score_range: Tuple[float] = (0.8, 1.0)
    min_2d_iou: Union[float, None] = 0.8

    min_sightings: int = 2
    max_t_no_sightings: int = 0.4
    mask_downsample_factor: int = 8
    min_max_extent: float = 0.25
    clustering_epsilon: float = 0.25
    plane_prune_params: Tuple[float] = (3.0, 3.0, 0.5)
    segment_graveyard_time: float = 15.0
    segment_graveyard_dist: float = 10.0
    iou_voxel_size: float = 0.2
    segment_voxel_size: float = 0.05
    segment_outlier_removal_std: float = 1.0
    
    def __post_init__(self):
        if self.semantic_association_method.lower() == 'none':
            self.semantic_association_method = None

    @classmethod
    def from_yaml(cls, yaml_path: str, run: str = None):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if run is not None and run in data:
            data = data[run]
        return cls(**data)

    def get_segment_params(self) -> 'SegmentParams':
        return SegmentParams(
            voxel_size=self.segment_voxel_size,
            outlier_removal_std=self.segment_outlier_removal_std
        )
    
@dataclass
class SegmentParams():

    voxel_size: float = 0.05
    outlier_removal_std: float = 1.0

    def __post_init__(self):
        if self.outlier_removal_std <= 0 or \
                np.isinf(self.outlier_removal_std):
            self.outlier_removal_std = None