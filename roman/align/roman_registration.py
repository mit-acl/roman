import numpy as np
from typing import List
from dataclasses import dataclass
from enum import Enum

import clipperpy

from roman.align.object_registration import ObjectRegistration
from roman.object.object import Object

class FusionMethod(Enum):
    GEOMETRIC_MEAN = clipperpy.invariants.ROMAN.GEOMETRIC_MEAN
    ARITHMETIC_MEAN = clipperpy.invariants.ROMAN.ARITHMETIC_MEAN
    PRODUCT = clipperpy.invariants.ROMAN.PRODUCT

@dataclass
class ROMANParams:
    
    point_dim: int = 3
    fusion_method = FusionMethod.GEOMETRIC_MEAN

    sigma: float = 0.4
    epsilon: float = 0.6
    mindist: float = 0.2

    gravity: bool = False
    volume: bool = False
    pca: bool = False
    extent: bool = False
    semantics_dim: int = 0
    gravity_unc_ang_rad: float = 0.0872665

    cos_min: float = 0.85
    cos_max: float = 1.0
    epsilon_shape: float = None
    pre_prune: bool = False

class ROMANRegistration(ObjectRegistration):
    def __init__(self, params: ROMANParams):
        super().__init__(dim=params.point_dim)

        ratio_feature_dim = 0
        self.volume = params.volume
        self.extent = params.extent
        self.pca = params.pca
        self.semantics = params.semantics_dim > 0
        self.pre_prune = params.pre_prune
        
        if self.pca:
            ratio_feature_dim += 3
        if self.volume:
            ratio_feature_dim += 1
        if self.extent:
            ratio_feature_dim += 3

        self.iparams = clipperpy.invariants.ROMANParams()
        self.iparams.point_dim = params.point_dim
        self.iparams.ratio_feature_dim = ratio_feature_dim
        self.iparams.cos_feature_dim = params.semantics_dim

        self.iparams.sigma = params.sigma
        self.iparams.epsilon = params.epsilon
        self.iparams.mindist = params.mindist

        self.iparams.distance_weight = 1.0
        self.iparams.ratio_weight = 1.0
        self.iparams.cosine_weight = 1.0

        self.iparams.ratio_epsilon = np.zeros(ratio_feature_dim) \
            if params.epsilon_shape is None \
            else np.ones(ratio_feature_dim) *  params.epsilon_shape 
        self.iparams.cosine_min = params.cos_min
        self.iparams.cosine_max = params.cos_max

        self.iparams.gravity_guided = params.gravity
        self.iparams.drift_aware = False
        
        if params.gravity:
            self.iparams.gravity_unc_ang_rad = params.gravity_unc_ang_rad
        
        return
    
    def _setup_clipper(self):
        invariant = clipperpy.invariants.ROMAN(self.iparams)
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, params)
        return clipper
    
    def _clipper_score_all_to_all(self, clipper, map1: List[Object], map2: List[Object]):

        if self.pre_prune:
            A_init = self._get_pruned_assoc(map1, map2)
        else:
            A_init = clipperpy.utils.create_all_to_all(len(map1), len(map2))

        map1_cl = np.array([self._object_to_clipper_list(p) for p in map1])
        map2_cl = np.array([self._object_to_clipper_list(p) for p in map2])
        self._check_clipper_arrays(map1_cl, map2_cl)

        clipper.score_pairwise_and_single_consistency(map1_cl.T, map2_cl.T, A_init)
        return clipper, A_init
    
    def _get_pruned_assoc(self, map1: List[Object], map2: List[Object]):
        A_all = clipperpy.utils.create_all_to_all(len(map1), len(map2))

        # prune based on semantics
        descriptors1 = np.array([p.semantic_descriptor.flatten() for p in map1]) # n1 x s
        descriptors2 = np.array([p.semantic_descriptor.flatten() for p in map2]) # n2 x s
        semantic_cos_sim = descriptors1 @ descriptors2.T # n1 x n2
        violates_cos_sim = semantic_cos_sim < self.iparams.cosine_min # n1 x n2
        to_delete = violates_cos_sim[A_all[:,0], A_all[:,1]] # n
        A_put = np.delete(A_all, to_delete, axis=0)
        

        # prune based on volume and pca
        shape_attrs1 = self._combined_shape_attributes(map1, A_put[:,0]) # n x 4
        shape_attrs2 = self._combined_shape_attributes(map2, A_put[:,1]) # n x 4
        shape_attrs_combined = np.stack([shape_attrs1, shape_attrs2], axis=2) # n x 4 x 2
        violates_ratio = (np.min(shape_attrs_combined, axis=2) / 
                          (np.max(shape_attrs_combined, axis=2) + 1e-6)) < self.iparams.ratio_epsilon # n x 4
        to_delete = np.any(violates_ratio, axis=1) # n
        A_put = np.delete(A_put, to_delete, axis=0)
        return A_put

    def _object_to_clipper_list(self, object: Object):        
        object_as_list = object.center.reshape(-1).tolist()[:self.dim]
        if self.pca:
            object_as_list += [object.linearity, object.planarity, object.scattering]
        if self.volume:
            object_as_list.append(object.volume)
        if self.extent:
            object_as_list += sorted(object.extent)
        if self.semantics:
            object_as_list += np.array(object.semantic_descriptor).tolist()
        return object_as_list 
    
    def _check_clipper_arrays(self, map1_cl, map2_cl):
        assert map1_cl.shape[1] == map2_cl.shape[1]
        # TODO: check that the number of point elements + feature elements is correct
        # if self.use_gravity:
        #     assert map1_cl.shape[1] == 3 + 2, f"map1_cl.shape[1] = {map1_cl.shape[1]}"
        return