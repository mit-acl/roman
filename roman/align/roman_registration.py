import numpy as np
from typing import List
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy
import shapely

import clipperpy

from roman.align.object_registration import ObjectRegistration
from roman.object.object import Object
from roman.map.global_nearest_neighbor import global_nearest_neighbor

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


class ROMANRegistration(ObjectRegistration):
    def __init__(self, params: ROMANParams):
        super().__init__(dim=params.point_dim)

        ratio_feature_dim = 0
        self.volume = params.volume
        self.extent = params.extent
        self.pca = params.pca
        self.semantics = params.semantics_dim > 0
        
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
    
    def permissive_associations(self, map1: List[Object], map2: List[Object], T: np.ndarray):
        map2_transformed = [deepcopy(obj) for obj in map2]
        [obj.transform(T) for obj in map2_transformed]
        
        def semantic_similarity_fun(o1, o2):
            cosine_sim = np.sum(o1.semantic_descriptor * o2.semantic_descriptor) / \
                (np.linalg.norm(o1.semantic_descriptor) * np.linalg.norm(o2.semantic_descriptor) + 1e-9)
            return cosine_sim
        
        def distance_similarity_fun(o1, o2):
            distance = np.linalg.norm(o1.center - o2.center)
            distance_score = 1 / (1 + distance)
            return distance_score
        
        max_dist = self.iparams.sigma
        dist_similarity_min = 1 / (1 + max_dist)
        semantic_similarity_min = self.iparams.cosine_min

        similarity_range = np.array([[semantic_similarity_min, 1.0], 
                                     [dist_similarity_min, 1.0]]).T
        similarity_fun = lambda o1, o2: np.array([
            semantic_similarity_fun(o1, o2),
            distance_similarity_fun(o1, o2)
        ])
        associations = global_nearest_neighbor(map1, map2_transformed, similarity_fun, similarity_range)
        
        # find convex hull of map1 and map2
        map1_convex_hull = shapely.geometry.MultiPoint([shapely.geometry.Point(obj.center.flatten()[:2]) for obj in map1]).convex_hull
        map2_convex_hull = shapely.geometry.MultiPoint([shapely.geometry.Point(obj.center.flatten()[:2]) for obj in map2_transformed]).convex_hull

        # find intersection of convex hulls
        intersection = map1_convex_hull.intersection(map2_convex_hull)
        
        # inflate intersection by sigma
        intersection = intersection.buffer(max_dist)
        
        # count number of objects in map1 and map2 that are in the intersection of the convex hulls
        num_objs_in_intersection = []
        for map_i in [map1, map2_transformed]:
            num_objs_in_intersection.append(0)
            for i, obj in enumerate(map_i):
                point = shapely.geometry.Point(obj.center.flatten()[:2])
                if intersection.contains(point):
                    num_objs_in_intersection[-1] += 1
                    
        min_num_objs_in_intersection = min(num_objs_in_intersection)

        return np.array(associations), min_num_objs_in_intersection

    def _setup_clipper(self):
        invariant = clipperpy.invariants.ROMAN(self.iparams)
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, params)
        return clipper
    
    def _clipper_score_all_to_all(self, clipper, map1: List[Object], map2: List[Object]):
        A_init = clipperpy.utils.create_all_to_all(len(map1), len(map2))

        map1_cl = np.array([self._object_to_clipper_list(p) for p in map1])
        map2_cl = np.array([self._object_to_clipper_list(p) for p in map2])
        self._check_clipper_arrays(map1_cl, map2_cl)

        clipper.score_pairwise_and_single_consistency(map1_cl.T, map2_cl.T, A_init)
        return clipper, A_init

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