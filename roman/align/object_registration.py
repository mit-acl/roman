import numpy as np
from typing import List
import matplotlib.pyplot as plt
import clipperpy
import logging
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

from roman.object.object import Object

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
        self.icp_max_correspondence_distance = 0.2
        self.icp_max_iterations = 50
        self.icp_min_fitness = 0.3

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

    def T_align(self, map1: List[Object], map2: List[Object], correspondences: np.array = None, xyz_yaw_only: bool = False):
        """
        Computes the transformation that aligns map2 to map1.

        Args:
            map1 (List[Object]): Object list in frame 1
            map2 (List[Object]): Object list in frame 2
            correspondences (np.array, shape=(n,2), optional): If correspondences have already
                been found, set to None. Otherwise, performs register before aligning. Aligns using
                Arun's method. Defaults to None.
            xyz_yaw_only (bool, optional): Fit only yaw + translation (gravity-aligned 3D maps). Defaults to False.

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

        return self._fit_transform(pts1, pts2, xyz_yaw_only)

    def _fit_transform(self, pts1, pts2, xyz_yaw_only=False):
        """Least-squares rigid transform from corresponding pts2 to pts1."""
        weights = np.ones((pts1.shape[0],1))
        weights = weights.reshape((-1,1))
        mean1 = (np.sum(pts1 * weights, axis=0) / np.sum(weights)).reshape(-1)
        mean2 = (np.sum(pts2 * weights, axis=0) / np.sum(weights)).reshape(-1)
        pts1_mean_reduced = pts1 - mean1
        pts2_mean_reduced = pts2 - mean2
        assert pts1_mean_reduced.shape == pts2_mean_reduced.shape
        H = pts1_mean_reduced.T @ (pts2_mean_reduced * weights)
        if xyz_yaw_only and self.dim == 3:
            yaw = np.arctan2(H[1, 0] - H[0, 1], H[0, 0] + H[1, 1])
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        else:
            U, s, Vh = np.linalg.svd(H)
            R = U @ Vh
            if np.allclose(np.linalg.det(R), -1.0):
                Vh_prime = Vh.copy()
                Vh_prime[-1,:] *= -1.0
                R = U @ Vh_prime
        t = mean1.reshape((-1,1)) - R @ mean2.reshape((-1,1))
        T = np.concatenate([np.concatenate([R, t], axis=1), np.hstack([np.zeros((1, R.shape[0])), [[1]]])], axis=0)
        return T

    def refine_with_icp(self, initial_transform: np.ndarray, point_cloud1: np.ndarray,
                        point_cloud2: np.ndarray, xyz_yaw_only: bool = False) -> np.ndarray:
        """Refine T_1_2 with point-to-point ICP (cloud2 is source, cloud1 is target).

        Nearest-neighbor pairs within icp_max_correspondence_distance are fitted
        iteratively using the same rigid/yaw-only estimator as object alignment.
        Keep the initial guess if there are fewer than three inliers, insufficient
        source-cloud fitness, a degenerate fit, or the inlier RMSE gets worse.
        """
        if self.dim != 3:
            raise ValueError("Submap point-cloud ICP requires dim=3")
        if point_cloud1 is None or point_cloud2 is None:
            logger.debug("Skipping ICP: submap point cloud is missing")
            return initial_transform

        clouds = []
        for cloud in (point_cloud1, point_cloud2):
            points = np.asarray(cloud, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("ICP point clouds must have shape (N, 3)")
            clouds.append(points[np.all(np.isfinite(points), axis=1)])
        target, source = clouds
        if min(len(target), len(source)) < 3:
            logger.debug("Skipping ICP: too few finite points")
            return initial_transform

        tree = cKDTree(target)

        def correspondences(T):
            transformed = source @ T[:3, :3].T + T[:3, 3]
            distances, indices = tree.query(transformed)
            mask = distances <= self.icp_max_correspondence_distance
            count = np.count_nonzero(mask)
            rmse = np.sqrt(np.mean(distances[mask] ** 2)) if count else np.inf
            return transformed[mask], target[indices[mask]], count, rmse

        T = initial_transform.copy()
        _, _, initial_count, initial_rmse = correspondences(T)
        for _ in range(self.icp_max_iterations):
            source_inliers, target_inliers, count, _ = correspondences(T)
            # A line or single point cannot determine a full rigid rotation.
            required_rank = 1 if xyz_yaw_only else 2
            axes = slice(0, 2) if xyz_yaw_only else slice(0, 3)
            if (count < 3 or 
                any(np.linalg.matrix_rank(points[:, axes] -
                points[:, axes].mean(axis=0)) < required_rank
                for points in (source_inliers, target_inliers))
            ):
                logger.debug("Skipping ICP: insufficient or degenerate correspondences")
                return initial_transform
            correction = self._fit_transform(target_inliers, source_inliers, xyz_yaw_only)
            T = correction @ T
            if np.allclose(correction, np.eye(4), atol=1e-6, rtol=0):
                break

        _, _, count, rmse = correspondences(T)
        fitness = count / len(source)
        if (not np.all(np.isfinite(T)) or count < 3 or fitness < self.icp_min_fitness or
                count < initial_count or rmse > initial_rmse + 1e-12):
            logger.debug("Keeping object alignment: ICP quality check failed")
            return initial_transform
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