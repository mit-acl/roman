import numpy as np
from typing import List
import matplotlib.pyplot as plt
import clipperpy

from roman.object.object import Object
from roman.object.segment import Segment

class InsufficientAssociationsException(Exception):
    
    def __init__(self, map1_len, map2_len, n_associations=None):
        self.map1_len = map1_len
        self.map2_len = map2_len
        self.n_associations = n_associations
        message = f"Insufficient associations. Map 1 length: {map1_len}. Map 2 length: {map2_len}. Associations: {n_associations}"
        super().__init__(message)

class ObjectRegistration():

    def __init__(self, dim=3):
        self.dim = dim

    def register(self, map1: List[Object], map2: List[Object]):
        if len(map1) == 0 or len(map2) == 0:
            return np.array([[]])
        clipper = self._setup_clipper()
        clipper, A_init = self._clipper_score_all_to_all(clipper, map1, map2)
        clipper.solve()
        Ain = clipper.get_selected_associations()
        return Ain
    
    def _setup_clipper(self):
        raise NotImplementedError
    
    def _object_to_clipper_list(self, object: Object):
        raise NotImplementedError
    
    def _check_clipper_arrays(self, map1_cl, map2_cl):
        return
    
    def _clipper_score_all_to_all(self, clipper, map1: List[Object], map2: List[Object]):
        A_init = clipperpy.utils.create_all_to_all(len(map1), len(map2))

        map1_cl = np.array([self._object_to_clipper_list(p) for p in map1])
        map2_cl = np.array([self._object_to_clipper_list(p) for p in map2])
        self._check_clipper_arrays(map1_cl, map2_cl)

        clipper.score_pairwise_consistency(map1_cl.T, map2_cl.T, A_init)
        return clipper, A_init
    
    def get_MCA(self, map1: List[Object], map2: List[Object]):
        clipper = self._setup_clipper()
        clipper, A_init = self._clipper_score_all_to_all(clipper, map1, map2)
        M = clipper.get_affinity_matrix()
        C = clipper.get_constraint_matrix()
        return M, C, A_init
    
    def mno_clipper(self, map1: List[Object], map2: List[Object], num_solutions=2):
        M, C, A = self.get_MCA(map1, map2)
        M_orig = M.copy()
        clipper = clipperpy.CLIPPER(clipperpy.invariants.PairwiseInvariant(), clipperpy.Params())
        solutions = []

        for k in range(num_solutions):
            clipper.set_matrix_data(M=M, C=C)
            clipper.solve()

            solution_nodes = clipper.get_solution().nodes
            Ain = np.zeros((len(solution_nodes), 2)).astype(np.int64)
            for i in range(len(solution_nodes)):
                Ain[i,:] = A[solution_nodes[i],:]
            
            u_sol = clipper.get_solution().u.copy()
            for i in range(u_sol.shape[0]):
                u_sol[i] = u_sol[i] if i in solution_nodes else 0.0
            if len(solution_nodes) == 0:
                score = 0
            else:
                score = u_sol.T @ M_orig @ u_sol / (u_sol.T @ u_sol)
            solutions.append((Ain.copy(), score))

            if k + 1 < num_solutions:
                row_indices, col_indices = np.meshgrid(solution_nodes, solution_nodes, indexing='ij')
                if len(row_indices) != 0 and len(col_indices) != 0:
                    M[row_indices,col_indices] = 0.0

        return solutions

    def T_align(self, map1: List[Object], map2: List[Object], correspondences: np.array = None):
        """
        Computes the transformation that aligns map2 to map1.

        Args:
            map1 (List[Object]): Object list in frame 1
            map2 (List[Object]): Object list in frame 2
            correspondences (np.array, shape=(n,2), optional): If correspondences have already 
                been found, set to None. Otherwise, performs register before aligning. Aligns using 
                Arun's method. Defaults to None.

        Returns:
            np.array: Transformation matrix that aligns map2 to map1
        """
        if len(map1) == 0 or len(map2) == 0:
            raise InsufficientAssociationsException(len(map1), len(map2))

        if correspondences is None:
            correspondences = self.register(map1, map2)
        if len(correspondences) < self.dim:
            raise InsufficientAssociationsException(len(map1), len(map2), len(correspondences))

        pts1 = np.array([map1[corr[0]].center.reshape(-1)[:self.dim] for corr in correspondences])
        pts2 = np.array([map2[corr[1]].center.reshape(-1)[:self.dim] for corr in correspondences])

        weights = np.ones((pts1.shape[0],1))
        weights = weights.reshape((-1,1))
        mean1 = (np.sum(pts1 * weights, axis=0) / np.sum(weights)).reshape(-1)
        mean2 = (np.sum(pts2 * weights, axis=0) / np.sum(weights)).reshape(-1)
        pts1_mean_reduced = pts1 - mean1
        pts2_mean_reduced = pts2 - mean2
        assert pts1_mean_reduced.shape == pts2_mean_reduced.shape
        H = pts1_mean_reduced.T @ (pts2_mean_reduced * weights)
        U, s, Vh = np.linalg.svd(H)
        R = U @ Vh
        if np.allclose(np.linalg.det(R), -1.0):
            Vh_prime = Vh.copy()
            Vh_prime[-1,:] *= -1.0
            R = U @ Vh_prime
        t = mean1.reshape((-1,1)) - R @ mean2.reshape((-1,1))
        T = np.concatenate([np.concatenate([R, t], axis=1), np.hstack([np.zeros((1, R.shape[0])), [[1]]])], axis=0)
        return T
    
    def view_registration(self, map1: List[Object], map2: List[Object], correspondences: np.array, T: np.array, ax=None, **kwargs):
        """
        Visualize the registration between map1 and map2

        Args:
            map1 (List[Object]): Object list in frame 1
            map2 (List[Object]): Object list in frame 2
            correspondences (np.array, shape=(n,2)): Correspondences between map1 and map2
            T (np.array): Transformation matrix that aligns map2 to map1
        """
        if ax is None:
            _, ax = plt.subplots()

        map2_cp = [obj.copy() for obj in map2]
        for obj in map2_cp:
            obj.transform(T)

        for obj in map1:
            obj.plot2d(ax, color='maroon', **kwargs)

        for obj in map2_cp:
            obj.plot2d(ax, color='blue', **kwargs)

        for corr in correspondences:
            ax.plot([map1[corr[0]].centroid[0], map2_cp[corr[1]].centroid[0]], 
                     [map1[corr[0]].centroid[1], map2_cp[corr[1]].centroid[1]], 
                     color='lawngreen', linestyle='dotted')
        
        ax.set_aspect('equal')
        return ax
    
    def _object_shape_attributes(self, object: Segment):
        return np.array([object.volume, object.linearity, 
                         object.planarity, object.scattering])
        
    def _combined_shape_attributes(self, object_map: List[Object], indices: np.ndarray):
        shape_attrs = np.array([self._object_shape_attributes(obj) for obj in object_map]) # n x 4
        return shape_attrs[indices, :]