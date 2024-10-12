import numpy as np

from dataclasses import dataclass, field
from typing import List
import os

import clipperpy
from roman.align.roman import ROMAN, ROMANParams
from roman.align.ransac_reg import RansacReg
from roman.align.dist_reg_with_pruning import DistRegWithPruning, GravityConstraintError

@dataclass
class SubmapAlignParams:
    dim: int = 3
    method: str = 'spvg'
    fusion_method: str = 'geometric_mean'
    submap_radius: float = 15.0
    submap_center_dist: float = 10.0
    submap_center_time: float = 30.0 # time threshold between segments and submap center times
    submap_max_size: int = 40
    single_robot_lc: bool = False
    single_robot_lc_time_thresh: float = 0.0
    
    # registration params
    sigma: float = 0.4
    epsilon: float = 0.6
    mindist: float = 0.2
    epsilon_shape: float = 0.0
    ransac_iter: int = int(1e6)
    cosine_min: float = 0.85
    cosine_max: float = 0.95
    semantics_dim: int = 768
    
    def get_object_registration(self):
        if self.fusion_method == 'geometric_mean':
            sim_fusion_method = clipperpy.invariants.ROMAN.GEOMETRIC_MEAN
        elif self.fusion_method == 'arithmetic_mean':
            sim_fusion_method = clipperpy.invariants.ROMAN.ARITHMETIC_MEAN
        elif self.fusion_method == 'product':
            sim_fusion_method = clipperpy.invariants.ROMAN.PRODUCT
            

        if self.method in ['standard', 'gravity', 'pcavolgrav', 'extentvolgrav', 'spvg', 'sevg']:
            roman_params = ROMANParams()
            roman_params.point_dim = self.dim
            roman_params.sigma = self.sigma
            roman_params.epsilon = self.epsilon
            roman_params.min_dist = self.mindist
            roman_params.fusion = sim_fusion_method

            roman_params.gravity = self.method in ['gravity', 'pcavolgrav', 'extentvolgrav', 'spvg', 'sevg']
            roman_params.volume = self.method in ['pcavolgrav', 'extentvolgrav', 'spvg', 'sevg']
            roman_params.extent = self.method in ['extentvolgrav', 'sevg']
            roman_params.pca = self.method in ['pcavolgrav', 'spvg']
            roman_params.cos_min = self.cosine_min
            roman_params.cos_max = self.cosine_max
            
            if self.method == 'standard':
                method_name = f'{self.dim}D Point CLIPPER'
            elif self.method == 'gravity':
                method_name = 'Gravity Guided CLIPPER'
                roman_params.gravity = True
            elif self.method == 'pcavolgrav':
                method_name = f'Gravity Guided PCA feature-based Volume Registration'
            elif self.method == 'extentvolgrav':
                method_name = f'Gravity Guided Extent-based Volume Registration'
            elif self.method == 'spvg':
                method_name = 'CLIP Semantic + PCA + Volume + Gravity'
                roman_params.semantics_dim = self.semantics_dim
            elif self.method == 'sevg':
                method_name = 'Semantic + Extent + Volume + Gravity'
                roman_params.semantics_dim = self.semantics_dim

            registration = ROMAN(roman_params)

        elif self.method == 'prunevol':
            method_name = f'{self.dim}D Volume-based Pruning'
            registration = DistRegWithPruning(sigma=self.sigma, epsilon=self.epsilon, mindist=self.mindist, dim=self.dim, volume_epsilon=self.epsilon_shape, use_gravity=False)
        elif self.method == 'prunevolgrav':
            method_name = f'Gravity Filtered Volume-based Pruning'
            registration = DistRegWithPruning(sigma=self.sigma, epsilon=self.epsilon, mindist=self.mindist, dim=self.dim, volume_epsilon=self.epsilon_shape, use_gravity=True)
        elif self.method == 'prunegrav':
            method_name = f'Gravity Filtered Pruning'
            registration = DistRegWithPruning(sigma=self.sigma, epsilon=self.epsilon, mindist=self.mindist, dim=self.dim, use_gravity=True)
        elif self.method == 'ransac':
            method_name = 'RANSAC'
            registration = RansacReg(dim=self.dim, max_iteration=self.ransac_iter)
        else:
            assert False, "Invalid method"
        return registration
        
    
@dataclass
class SubmapAlignInputOutput:
    inputs: List[any]
    output_dir: str
    run_name: str
    input_type_pkl: bool = True
    input_type_json: bool = False
    input_gt_pose_yaml: List[str] = field(default_factory=lambda: [None, None])
    robot_names: List[str] = field(default_factory=lambda: ["0", "1"])
    lc_association_thresh: int = 4
    g2o_t_std: float = 1.0
    g2o_r_std: float = np.deg2rad(2.0)
    debug_show_maps: bool = False
    
    @property
    def output_img(self):
        return os.path.join(self.output_dir, f'{self.run_name}.png')
    
    @property
    def output_matrix(self):
        return os.path.join(self.output_dir, f'{self.run_name}.matrix.pkl')
    
    @property
    def output_viz(self):
        return os.path.join(self.output_dir, f'{self.run_name}.viz.pkl')
    
    @property
    def output_timing(self):
        return os.path.join(self.output_dir, f'{self.run_name}.timing.txt')
    
    @property
    def output_params(self):
        return os.path.join(self.output_dir, f'{self.run_name}.params.txt')
    
    @property
    def output_g2o(self):
        return os.path.join(self.output_dir, f'{self.run_name}.g2o')
    
    @property
    def output_lc_json(self):
        return os.path.join(self.output_dir, f'{self.run_name}.json')
    
    @property
    def output_submaps(self):
        return [os.path.join(self.output_dir, f'{self.run_name}.{rn}.sm.json') for rn in self.robot_names]