###########################################################
#
# similarity_metrics.py
#
# Classes with methods to calculate 
#
# Authors: Qingyuan Li
#
# January 27. 2025
#
###########################################################

import numpy as np
from typing import Tuple

class Wasserstein():

    @classmethod
    def principle_square_root(cls, A):
        """
        Compute the principle square root of a symmetric positive semi-definite matrix A

        - Source: https://en.wikipedia.org/wiki/Square_root_of_a_matrix#Positive_semidefinite_matrices
            - see "Solutions in close form/By diagonalization"
        - using Eigendecomposition: V D^{1/2} V^T * (V D^{1/2} V^T) = V D V^T = A
        """
        eigvals, eigvecs = np.linalg.eigh(A)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    @classmethod
    def wasserstein_metric(cls, gaussian_1: Tuple[np.ndarray, np.ndarray], gaussian_2: Tuple[np.ndarray, np.ndarray]):
        """
        Compute the Wasserstein metric between two Gaussian distributions

        - Source: https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions

        Args:
            gaussian_1 (Tuple[np.ndarray, np.ndarray]): mean and covariance of the first Gaussian distribution
            gaussian_2 (Tuple[np.ndarray, np.ndarray]): mean and covariance of the second Gaussian distribution
        """
        mu1, sigma1 = gaussian_1
        mu2, sigma2 = gaussian_2
        sigma2_sqrt = cls.principle_square_root(sigma2)
        return np.linalg.norm(mu1 - mu2) + np.trace(sigma1 + sigma2 - 2 * cls.principle_square_root(sigma2_sqrt @ sigma1 @ sigma2_sqrt))
    
class ChamferDistance():

    @classmethod
    def chamfer_distance(cls, pcd1, pcd2):
        """
        See [1] https://github.com/UM-ARM-Lab/Chamfer-Distance-API and [2] https://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Point-Cloud-Distance.
        
        The champer distance from pcd1 to pcd2 is the average of the distances from each point in pcd1 to its nearest point in pcd2.
            - o3d.geometry.PointCloud.compute_point_cloud_distance [2] returns an array of the distances for each point in the calling pointcloud to
              the other pointcloud, so we take the mean to get chamfer distance as a single metric.
        
        Instead of adding the directional champer distances like [1], we take the minimum, as we want to measure overlap and de-value extent. 
        The champer distance from a small pointcloud to a large enclosing pointcloud will be small, but this is not true in the other direction.

        Args:
            pcd1 (o3d.geometry.PointCloud): first point cloud
            pcd2 (o3d.geometry.PointCloud): second point cloud
        """
        if not pcd1.has_points() or not pcd2.has_points():
            return np.inf
        return min(np.mean(pcd1.compute_point_cloud_distance(pcd2)), np.mean(pcd2.compute_point_cloud_distance(pcd1)))
    
    @classmethod
    def norm_chamfer_distance(cls, pcd1, pcd2):
        """
        Compute the normalized chamfer distance between two point clouds.
        
        Normalization is done by dividing the chamfer distance by the 
        diagonal of the axis-aligned bounding box that contains both point clouds.
        
        Args:
            pcd1 (o3d.geometry.PointCloud): first point cloud
            pcd2 (o3d.geometry.PointCloud): second point cloud
        """
        chamfer_dist = cls.chamfer_distance(pcd1, pcd2)
        aabb1 = pcd1.get_axis_aligned_bounding_box()
        aabb2 = pcd2.get_axis_aligned_bounding_box()
        merged_min_bound = np.minimum(aabb1.min_bound, aabb2.min_bound)
        merged_max_bound = np.maximum(aabb1.max_bound, aabb2.max_bound) # TODO: try OBB from pointcloud
        # of OBB vertices instead
        merged_spread = merged_max_bound - merged_min_bound
        diag = np.linalg.norm(merged_spread)
        return 1 - (chamfer_dist / diag) if diag > 0 else 1.0