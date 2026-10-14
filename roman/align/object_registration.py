import numpy as np
from typing import List
import matplotlib.pyplot as plt
import clipperpy
import logging
import open3d as o3d

logger = logging.getLogger(__name__)

from roman.object.object import Object
from roman.utils import transform_rm_roll_pitch

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
        self.icp_correspondence_distances = [1.0, 0.5, 0.25, 0.1]
        self.icp_max_iterations = 30
        self.icp_min_fitness = 0.2
        self.icp_max_source_points = 10000
        self.icp_normal_radius = 0.4
        self.icp_max_translation = 1.5
        self.icp_max_rotation_deg = 10.0

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

    def prepare_icp_cloud(self, points: np.ndarray):
        """Open3D cloud (finite points, normals estimated) to use as an ICP target.

        Normals are what point-to-plane ICP needs from the target; computing them is the
        expensive part, so build this once per submap and pass it to refine_with_icp as
        target_pcd for every pair that uses that submap as the target. Returns None if the
        cloud is missing or has fewer than 3 finite points.
        """
        if points is None:
            return None
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("ICP point clouds must have shape (N, 3)")
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < 3:
            return None
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.icp_normal_radius, max_nn=30))
        return pcd

    def refine_with_icp(self, initial_transform: np.ndarray, point_cloud1: np.ndarray,
                        point_cloud2: np.ndarray, xyz_yaw_only: bool = False,
                        target_pcd=None) -> np.ndarray:
        """Refine T_1_2 with point-to-plane ICP (cloud2 is source, cloud1 is target).

        Open3D point-to-plane ICP with a Tukey robust loss, run coarse to fine over
        icp_correspondence_distances (each stage starts from the previous result; the
        larger early distances let it pull in from ROMAN's typical 0.2-1 m errors). With
        xyz_yaw_only, roll and pitch are removed after every stage (submap frames are
        gravity aligned). The source is randomly subsampled to icp_max_source_points.

        The initial guess is kept unless the result is finite, its fitness at the finest
        distance (share of source points with a target within it) is at least
        icp_min_fitness and not lower than the initial guess's, and it moves the initial
        guess by at most icp_max_translation (m) and icp_max_rotation_deg (point-to-plane
        ICP can slide or diverge on scenes that constrain it poorly, e.g. mostly ground).

        target_pcd, if given, is prepare_icp_cloud(point_cloud1); pass one to reuse the
        target normals across calls that share a target cloud.
        """
        if self.dim != 3:
            raise ValueError("Submap point-cloud ICP requires dim=3")
        target = target_pcd if target_pcd is not None else self.prepare_icp_cloud(point_cloud1)
        if target is None or point_cloud2 is None:
            logger.debug("Skipping ICP: submap point cloud is missing or too small")
            return initial_transform
        source = np.asarray(point_cloud2, dtype=float)
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("ICP point clouds must have shape (N, 3)")
        source = source[np.all(np.isfinite(source), axis=1)]
        if len(source) < 3:
            logger.debug("Skipping ICP: too few finite source points")
            return initial_transform
        if self.icp_max_source_points is not None and len(source) > self.icp_max_source_points:
            # fixed seed so results are repeatable
            keep = np.random.default_rng(0).choice(len(source), self.icp_max_source_points, replace=False)
            source = source[keep]
        source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))

        finest = min(self.icp_correspondence_distances)
        reg = o3d.pipelines.registration
        initial_fitness = reg.evaluate_registration(source, target, finest, initial_transform).fitness
        criteria = reg.ICPConvergenceCriteria(max_iteration=self.icp_max_iterations)
        T = np.array(initial_transform, dtype=float)
        for distance in sorted(self.icp_correspondence_distances, reverse=True):
            estimation = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=distance))
            T = reg.registration_icp(source, target, distance, T, estimation, criteria).transformation
            if xyz_yaw_only:
                T = transform_rm_roll_pitch(T)
        fitness = reg.evaluate_registration(source, target, finest, T).fitness

        correction = T @ np.linalg.inv(initial_transform)
        moved = np.linalg.norm(correction[:3, 3])
        turned = np.rad2deg(np.arccos(np.clip((np.trace(correction[:3, :3]) - 1) / 2, -1, 1)))
        if (not np.all(np.isfinite(T)) or fitness < self.icp_min_fitness or fitness < initial_fitness or
                moved > self.icp_max_translation or turned > self.icp_max_rotation_deg):
            logger.debug(f"Keeping object alignment: ICP check failed (fitness {initial_fitness:.2f} -> "
                         f"{fitness:.2f}, moved {moved:.2f} m, {turned:.1f} deg)")
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