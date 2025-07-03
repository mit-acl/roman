import numpy as np
from typing import List
from dataclasses import dataclass
from enum import Enum
from sklearn.neighbors import NearestNeighbors

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
    semantics_dim = 0

    cos_min: float = 0.85
    cos_max: float = 0.95
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
        
        self.k = 1
        return
    
    def _setup_clipper(self):
        invariant = clipperpy.invariants.ROMAN(self.iparams)
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, params)
        return clipper
    
    def _clipper_score_all_to_all(self, clipper, map1: List[Object], map2: List[Object]):
        # A_init = clipperpy.utils.create_all_to_all(len(map1), len(map2))
        map1_features = np.array([p.semantic_descriptor.tolist() + [p.linearity(), p.planarity(), p.scattering()] for p in map1])
        map2_features = np.array([p.semantic_descriptor.tolist() + [p.linearity(), p.planarity(), p.scattering()] for p in map2])
        
        map1_indices = self._knn(map1_features, map2_features)
        map2_indices = self._knn(map2_features, map1_features)
        pairs_set = set()
        for i, idx in enumerate(map1_indices):
            for j in idx:
                pairs_set.add((i, j))
        for j, idx in enumerate(map2_indices):
            for i in idx:
                pairs_set.add((i, j))
        print("PAIRS SET SIZE: ", len(pairs_set))
        print(pairs_set[:5,:])
        A_put = np.array(list(pairs_set)).astype(np.int32)


        map1_cl = np.array([self._object_to_clipper_list(p) for p in map1])
        map2_cl = np.array([self._object_to_clipper_list(p) for p in map2])
        self._check_clipper_arrays(map1_cl, map2_cl)
        print(map1_cl[0,:])
        print(f"cosine similarity")
        print(np.sum(map1_cl[A_put[0,0],3:].reshape(-1) * map2_cl[A_put[0,1],3:].reshape(-1)))

        print(map1_cl.T.shape, map2_cl.T.shape, A_put.shape)
        print(map1_cl.dtype, map2_cl.dtype, A_put.dtype)
        print(A_put)
        clipper.score_pairwise_and_single_consistency(map1_cl.T, map2_cl.T, A_put)
        M = clipper.get_affinity_matrix()
        print(M)
        C = clipper.get_constraint_matrix()
        print(C)
        return clipper, A_put

    def _object_to_clipper_list(self, object: Object):        
        object_as_list = object.center.reshape(-1).tolist()[:self.dim]
        if self.pca:
            e = object.normalized_eigenvalues()
            object_as_list += [object.linearity(e), object.planarity(e), object.scattering(e)]
        if self.volume:
            object_as_list.append(object.volume)
        if self.extent:
            object_as_list += sorted(object.extent)
        if self.semantics:
            object_as_list += (np.array(object.semantic_descriptor) / np.linalg.norm(np.array(object.semantic_descriptor))).tolist()
        return object_as_list 
    
    def _check_clipper_arrays(self, map1_cl, map2_cl):
        assert map1_cl.shape[1] == map2_cl.shape[1]
        # TODO: check that the number of point elements + feature elements is correct
        # if self.use_gravity:
        #     assert map1_cl.shape[1] == 3 + 2, f"map1_cl.shape[1] = {map1_cl.shape[1]}"
        return
    
    def _knn(self, query, reference):
        """
        Find the k nearest neighbors of each point in query in reference
        query: np.array of shape (n, d)
        reference: np.array of shape (m, d)
        """
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='kd_tree').fit(reference)
        _, indices = nbrs.kneighbors(query) # dimensions: (n, k)
        return indices
