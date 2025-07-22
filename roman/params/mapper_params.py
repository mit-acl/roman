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

from dataclasses import dataclass
import yaml
from typing import Tuple

@dataclass
class MapperParams():
    """
    Params for ROMAN open-set segment mapper object.
    
    Args:
        geometric_local_assoication_method (str): geometric method for associating segments across frames.
            ('chamfer', 'iou, 'iom') for chamfer distance, voxel IOU, or voxel Intersection-over-Minimum (IOM).
        semantic_local_association_method (str): semantic method for associating segments across frames.
            ('cosine_similarity', 'none') for cosine similarity or no semantic association.
        association_method_combination (str): method for combining geometric and semantic association scores (normalized to 0-1).
            ('gm', 'am', 'prod', 'max') for geometric mean, arithmetic mean, product, or max
        min_association_score (float): minimum association score (geometric and semantic combined, normalized
            to 0-1) for frame-to-frame association or merging objects. 
        min_sightings (int): minimum number of sightings to consider an object
        max_t_no_sightings (int): maximum time without a sighting before moving 
            an object to inactive
        mask_downsample_factor (int): factor by which to downsample the mask
        min_max_extent (float): to get rid of very small objects
        plane_prune_params (Tuple[float]): to get rid of planar objects (likely background)
        segment_graveyard_time (float): time after which an inactive segment is sent to the graveyard
        segment_graveyard_dist (float): distance traveled after which an inactive segment is sent to the graveyard
        iou_voxel_size (float): voxel size for IOU calculation (used if geometric_local_assoication_method is 'iou' or 'iom')
        segment_voxel_size (float): voxel size for segment representation (point-cloud downsampling)

    Returns:
        MapperParams: params object
    """

    geometric_local_assoication_method: str = 'iou'
    semantic_local_association_method: str = 'cosine_similarity'
    association_method_combination: str = 'gm'
    min_association_score: float = 0.6

    # association_method: str = 'iou'
    # merge_method: str = 'iou'
    # iom_as_iou: bool = False

    # min_iou: float = 0.25
    # max_chamfer_dist: float = 0.5
    min_sightings: int = 2
    max_t_no_sightings: int = 0.4
    # merge_objects_iou_3d: float = 0.25
    # merge_objects_iou_2d: float = 0.8
    # merge_objects_max_chamfer_dist: float = 0.5
    mask_downsample_factor: int = 8
    min_max_extent: float = 0.25
    clustering_epsilon: float = 0.25
    plane_prune_params: Tuple[float] = (3.0, 3.0, 0.5)
    segment_graveyard_time: float = 15.0
    segment_graveyard_dist: float = 10.0
    iou_voxel_size: float = 0.2
    segment_voxel_size: float = 0.05
    
    def __post_init__(self):
        if self.semantic_local_association_method.lower() == 'none':
            self.semantic_local_association_method = None

    @classmethod
    def from_yaml(cls, yaml_path: str, run: str = None):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if run is not None and run in data:
            data = data[run]
        return cls(**data)