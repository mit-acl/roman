import numpy as np
from numpy.linalg import norm
import cv2 as cv
from typing import List, Tuple
import shapely
from dataclasses import dataclass

from robotdatapy.data.img_data import CameraParams
from robotdatapy.transform import transform, aruns
from robotdatapy.camera import xyz_2_pixel, pixel_depth_2_xyz

import open3d as o3d
from roman.map.observation import Observation 
from roman.map.voxel_grid import VoxelGrid
from roman.object.object import Object


class SegmentMinimalData(Object):
    
    # TODO: Handle center ref for segment minimal data
    
    def __init__(
        self,
        id: int,
        center: np.array,
        gaussian: Tuple[np.array, np.array],
        volume: float,
        linearity: float,
        planarity: float,
        scattering: float,
        extent: np.array,
        semantic_descriptor: np.array,
        first_seen: float,
        last_seen: float
    ):
        super().__init__(center, 3, id, volume=volume)
        self._gaussian = gaussian # no longer dynamic
        self._linearity = linearity
        self._planarity = planarity
        self._scattering = scattering
        self.extent = extent
        self.semantic_descriptor = semantic_descriptor
        self.first_seen = first_seen
        self.last_seen = last_seen
        
    def normalized_eigenvalues(self):
        return None
    
    def linearity(self, e=None):
        return self._linearity
        
    def planarity(self, e=None):
        return self._planarity

    def scattering(self, e=None):
        return self._scattering
    
    def reference_time(self, use_avg_time=True):
        if not use_avg_time: return self.first_seen
        else: return (self.first_seen + self.last_seen) / 2.0
    
    @property
    def gaussian(self):
        return self._gaussian

class Segment(Object):

    # TODO: separate from observation and from points class
    def __init__(self, observation: Observation, camera_params: CameraParams, 
                 id: int = 0, voxel_size: float = 0.05):
        # initialize parent class
        super().__init__(centroid=np.zeros(3), dim=3, id=id)
        self.observations = [observation.copy(include_mask=False)]
        self.first_seen = observation.time
        self.last_seen = observation.time
        self.camera_params = camera_params
        self.num_sightings = 1
        self.edited = True
        self.last_observation = observation
        self.points = None
        self.voxel_size = voxel_size  # voxel size used for maintaining point clouds
        self.voxel_grid = dict()
        self.last_propagated_mask = None
        self.last_propagated_time = None
        self.semantic_descriptor = None
        self.semantic_descriptor_cnt = 0
        self._center_ref = "mean" # TODO: make enum. For now: mean or bottom-middle

        # memoized attributes
        self._obb = None
        self._pcd = None
        self._aabb = None
        self._gaussian = None
        self._eigvals = None
        self._mask = None
        self._mask_last_pose = np.nan
        self._pixels = None
        self._pixels_past_pose = np.nan
        self._convex_hull = None
        self._convex_hull_last_pose = np.nan
        
        self._integrate_points_from_observation(observation)

    def update(self, observation: Observation, integrate_points=True):
        """Update a 3D segment with a new observation

        Args:
            observation (Observation): Input observation object
            force (bool, optional): If true, do not attempt 3D reconstruction. Defaults to False.
            integrate_points (bool, optional): If true, integrate point cloud contained in observation. Defaults to True.
        """

        # See if this will break the reconstruction
        # if not force:
        #     try: 
        #         self.reconstruction_from_observations(self.observations + [observation], width_height=False)
        #     except:
        #         return
        # self.reset_memoized()
            
        # Integrate point measurements
        if integrate_points:
            self._integrate_points_from_observation(observation)
            if observation.clip_embedding is not None:
                self._add_semantic_descriptor(observation.clip_embedding)

        self.num_sightings += 1
        self.observations.append(observation.copy(include_mask=False))

        if observation.time > self.last_seen:
            self.last_seen = observation.time
            self.last_observation = observation.copy(include_mask=True)

    def update_from_segment(self, segment):
        for obs in segment.observations:
            # none of the observations will have masks, so need to update with 
            # the last_observation copy (which will have a mask) instead
            if obs.time == segment.last_seen:
                obs = segment.last_observation
            self.update(obs, integrate_points=False)
        self._integrate_points_from_segment(segment)
        if segment.semantic_descriptor is not None:
            self._add_semantic_descriptor(segment.semantic_descriptor, segment.semantic_descriptor_cnt)
    
    def _integrate_points_from_observation(self, observation: Observation):
        """Integrate point cloud in the input observation object

        Args:
            observation (Observation): input observation object
        """
        # self.reset_memoized()

        if observation.point_cloud is None:
            return
        # Convert observed points to global frame
        Twb = observation.pose
        Rwb = Twb[:3,:3]
        twb = Twb[:3,3].reshape((3,1))
        points_obs_body = observation.point_cloud.T
        num_points_obs = points_obs_body.shape[1]
        points_obs_world = Rwb @ points_obs_body + np.repeat(twb, num_points_obs, axis=1)
        self._add_points(points_obs_world.T)

    def _integrate_points_from_segment(self, segment):
        """Integrate the points from another segment into this segment.
        Both segments are assumed to be in the same reference frame

        Args:
            segment (Segment): _description_
        """
        # self.reset_memoized() # reset bbox
        if segment.num_points > 0:
            self._add_points(segment.points)
        else: # TODO: not sure how this is reached?
            self.reset_memoized()
            self.points = segment.points

    def _add_points(self, points):
        assert points.shape[1] == 3
        if points.shape[0] == 0:
            return
        
        self.reset_memoized() # reset memoized

        if self.points is None:
            self.points = points
        else:
            self.points = np.concatenate((self.points, points), axis=0)

        self._cleanup_points()
    
    def _cleanup_points(self):
        if self.points is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd_sampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            pcd_pruned, _ = pcd_sampled.remove_statistical_outlier(10, 1.0)

            if pcd_pruned.is_empty():
                self.points = None
            else:
                self.points = np.asarray(pcd_pruned.points) 

            self._pcd = pcd_pruned # memoize
                
    def final_cleanup(self, epsilon=0.25, min_points=10):
        """
        Performs DBSCAN clustering on the points of the segment and returns the largest cluster

        Args:
            epsilon (float, optional): Max distance between two samples to be eligible to be in same cluster. Defaults to 0.25.
            min_points (int, optional): Number of points needed to form a cluster. Defaults to 10.
        """
        if self.points is not None:
            # Perform DBSCAN clustering
            labels = np.array(self.pcd.cluster_dbscan(eps=epsilon, min_points=min_points))

            # Number of clusters, ignoring noise if present
            max_label = labels.max()
            
            # get largest cluster
            cluster_sizes = np.zeros(max_label + 1)
            for i in range(max_label + 1):
                cluster_sizes[i] = np.sum(labels == i)
            max_cluster = np.argmax(cluster_sizes)

            # Filter out any points not belonging to max cluster
            filtered_indices = np.where(labels == max_cluster)[0]
            self.points = self.points[filtered_indices]

            self.reset_memoized() # clear memory-intensive attributes
               

    @property
    def num_points(self):
        if self.points is None:
            return 0
        else:
            return self.points.shape[0]
        
    def reset_memoized(self):
        self._aabb = None
        self._obb = None
        self._pcd = None
        self._gaussian = None
        self._eigvals = None
        self._mask = None
        self._mask_last_pose = np.nan
        self._pixels = None
        self._pixels_past_pose = np.nan
        self._convex_hull = None
        self._convex_hull_last_pose = np.nan
        self.voxel_grid = dict()

    @property
    def obb(self):
        if self._obb is None:
            self._obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                            o3d.utility.Vector3dVector(self.points))
        return self._obb
        
    @property
    def volume(self):
        if self.num_points > 4: # 4 is the minimum number of points needed to define a 3D box
            return self.obb.volume()
        else:
            return 0.0
        
    @property
    def extent(self):
        if self.num_points > 4:
            return self.obb.extent
        else:
            return np.zeros(3)
        
    @property
    def center(self):
        if self._center_ref == 'bottom_middle':
            pt = np.median(self.points, axis=0)
            pt[2] = np.min(self.points[:,2])
            return pt
        elif self._center_ref == 'mean':
            return self.pcd.get_center().reshape(self.dim, 1)
        else:
            assert False, "Invalid center reference point type"
            
    def reference_time(self, use_avg_time=True):
        if not use_avg_time: return self.first_seen
        else: return (self.first_seen + self.last_seen) / 2.0
        
    def get_voxel_grid(self, voxel_size: float) -> VoxelGrid:
        if self.num_points > 0:
            if voxel_size not in self.voxel_grid:
                self.voxel_grid[voxel_size] = VoxelGrid.from_points(self.points, voxel_size)
            return self.voxel_grid[voxel_size]
        raise ValueError("No points in segment")
    
    @property
    def aabb(self):
        if self._aabb is None:
            self._aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(self.points))
        
    def aabb_volume(self):
        """Return the volume of the 3D axis-aligned bounding box
        """
        if self.num_points > 0:
            return self.aabb.volume()
        return 0.0

    @property
    def last_mask(self):
        if self.last_observation.mask is not None:
            return self.last_observation.mask.copy()
        else:
            return None
    
    def reconstruct_mask(self, pose, downsample_factor=1):
        if self._mask is None or not np.allclose(pose, self._mask_last_pose):
            mask = np.zeros((self.camera_params.height, self.camera_params.width), dtype=np.uint8)

            bbox = self.reprojected_bbox(pose)
            if bbox is not None:
                upper_left, lower_right = bbox
                mask[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]] = 1

            if downsample_factor == 1:
                return mask.astype('uint8')
            
            # Additional downsampling
            mask = np.array(cv.resize(
                        mask,
                        (mask.shape[1]//downsample_factor, mask.shape[0]//downsample_factor), 
                        interpolation=cv.INTER_NEAREST
                    )).astype('uint8')
            self._mask = mask
            self._mask_last_pose = pose

        return self._mask
        
    def _pixels_2d(self, pose):
        if self._pixels is None or not np.allclose(pose, self._pixels_past_pose):
            if self.points is None:
                return None
            points_c = transform(np.linalg.inv(pose), self.points, axis=0)
            points_c = points_c[points_c[:,2] >= 0]
            if len(points_c) == 0:
                return None
            pixels = xyz_2_pixel(points_c, self.camera_params.K)
            pixels = pixels[np.bitwise_and(pixels[:,0] >= 0, pixels[:,0] < self.camera_params.width), :]
            pixels = pixels[np.bitwise_and(pixels[:,1] >= 0, pixels[:,1] < self.camera_params.height), :]
            self._pixels = pixels
            self._pixels_past_pose = pose

        if len(self._pixels) == 0:
            return None
        return self._pixels
    
    def reprojected_bbox(self, pose):
        pixels = self._pixels_2d(pose)
        if pixels is None:
            return None
        upper_left = np.max([np.min(pixels, axis=0).astype(int), [0, 0]], axis=0)
        lower_right = np.min([np.max(pixels, axis=0).astype(int), 
                                [self.camera_params.width, self.camera_params.height]], axis=0)
        # if width == 0 or height == 0
        if lower_right[0] - upper_left[0] <= 0 or lower_right[1] - upper_left[1] <= 0:
            return None
        return upper_left, lower_right
    
    def propagated_last_mask(self, t, pose, downsample_factor=1):
        if self.last_propagated_mask is not None and self.last_propagated_time == t:
            return self.last_propagated_mask
        
        if self.last_observation.mask_downsampled is None:
            self.last_propagated_mask = None
            self.last_propagated_time = t
            return None
        
        if self.points is None:
            return self.last_observation.mask_downsampled
        
        # get depth of the segment
        points_all_cm1 = transform(np.linalg.inv(self.last_observation.pose), self.points, axis=0)
        depth = np.mean(points_all_cm1[:,2])
        mask_unchanged = self.last_observation.mask_downsampled
        
        # compute the bounding box of the segment for the last observation camera pose
        bbox_ul = np.array(np.min(np.argwhere(mask_unchanged > 0), axis=0))
        bbox_lr = np.array(np.max(np.argwhere(mask_unchanged > 0), axis=0))
        bbox_ul *= downsample_factor
        bbox_lr *= downsample_factor
        bbox_ur = np.array([bbox_lr[0], bbox_ul[1]])
        bbox_ll = np.array([bbox_ul[0], bbox_lr[1]])
        points_uvm1 = np.array([bbox_ul, bbox_ur, bbox_lr, bbox_ll])
        
        # compute the 3D points of the bounding box in the last observation camera frame
        points_cm1 = np.array([pixel_depth_2_xyz(p[0], p[1], depth, self.camera_params.K) for p in points_uvm1])

        # get corresponding world coordinates for the bounding box points
        points_w = transform(self.last_observation.pose, points_cm1, axis=0)
        
        # project the bounding box points to the current camera frame
        points_c = transform(np.linalg.inv(pose), points_w, axis=0)
        points_uv = xyz_2_pixel(points_c, self.camera_params.K, axis=0)
        points_uvm1 = np.array([pt / downsample_factor for pt in points_uvm1])
        points_uv = np.array([pt / downsample_factor for pt in points_uv])
        T_pixels = aruns(points_uv, points_uvm1)
        mask = cv.warpAffine(mask_unchanged.astype(np.float32), T_pixels[:2, :3], (mask_unchanged.shape[1], mask_unchanged.shape[0])) 
                            # flags=cv.INTER_NEAREST)
        mask = mask.astype(np.uint8)
        
        self.last_propagated_mask = mask
        self.last_propagated_time = t
        return mask
    
    def outline_2d(self, pose):
        if self._convex_hull is None or not np.allclose(pose, self._convex_hull_last_pose):
            pixels = self._pixels_2d(pose)
            if pixels is None:
                return None
            convex_hull = shapely.convex_hull(shapely.MultiPoint(pixels))
            self._convex_hull = convex_hull
            self._convex_hull_last_pose = pose

        if type(self._convex_hull) == shapely.Polygon:
            return np.array(self._convex_hull.exterior.coords)
        elif type(self._convex_hull) == shapely.LineString:
            return np.array(self._convex_hull.coords)
        
    @property
    def pcd(self):
        if self._pcd is None:
            self._pcd = o3d.geometry.PointCloud()
            self._pcd.points = o3d.utility.Vector3dVector(self.points)
        return self._pcd

    @property 
    def gaussian(self):
        if self._gaussian is None:
            self._gaussian = self.pcd.compute_mean_and_covariance()
        return self._gaussian
        
    @property
    def normalized_eigenvalues(self):
        """Compute the normalized eigenvalues of the covariance matrix
        as a np array [e1, e2, e3]
        e1 >= e2 >= e3 so that the sum is one
        """
        if self._eigvals is None:
            _, C = self.gaussian
            _, eigvals, _ = np.linalg.svd(C)  # svd return in descending order
            self._eigvals = eigvals / eigvals.sum()
        return self._eigvals

    def linearity(self, e: np.ndarray=None):
        """ Large if similar to a 1D line (Weinmann et al. ISPRS 2014)

        Args:
            e (np.ndarray): normalized eigenvalues of this point cloud
        """
        if e is None:
            e = self.normalized_eigenvalues
        return (e[0]-e[1]) / e[0]

    def planarity(self, e: np.ndarray=None):
        """ Large if similar to a 2D plane (Weinmann et al. ISPRS 2014)
        Args:
            e (np.ndarray): normalized eigenvalues of this point cloud
        """
        if e is None:
            e = self.normalized_eigenvalues
        return (e[1]-e[2]) / e[0]

    def scattering(self, e: np.ndarray=None):
        """Large if this object is 3D, i.e., neither a line nor a plane (Weinmann et al. ISPRS 2014)

        Args:
            e (np.ndarray): normalized eigenvalues of this point cloud
        """
        if e is None:
            e = self.normalized_eigenvalues
        return e[2] / e[0]
    
    def _add_semantic_descriptor(self, descriptor: np.ndarray, cnt: int = 1):
        if self.semantic_descriptor is None:
            assert cnt == 1, "Multiple Initialization of Semantic Descriptor"
            self.semantic_descriptor = descriptor / norm(descriptor)
            self.semantic_descriptor_cnt = cnt
        else:
            self.semantic_descriptor = (
                self.semantic_descriptor * self.semantic_descriptor_cnt
                / (self.semantic_descriptor_cnt + cnt) 
                + descriptor * cnt / norm(descriptor)
                / (self.semantic_descriptor_cnt + cnt)
            )
            self.semantic_descriptor_cnt += cnt
        self.semantic_descriptor /= norm(self.semantic_descriptor) # renormalize
        
    def transform(self, T):
        if self.points is not None:
            self.points = transform(T, self.points, axis=0)
            self.reset_memoized()
            
    def minimal_data(self):
        e = self.normalized_eigenvalues
        return SegmentMinimalData(
            self.id,
            self.center,
            self.gaussian,
            self.volume,
            self.linearity(e),
            self.planarity(e),
            self.scattering(e),
            self.extent,
            self.semantic_descriptor,
            self.first_seen,
            self.last_seen
        )
            
    @property
    def viz_color(self):
        np.random.seed(self.id)
        color = np.random.randint(0, 255, 3).tolist()
        return color
        
    @classmethod
    def from_pickle(cls, data):
        return data
        # new_obj = cls(
        #     data["centroid"],
        #     data["rot_mat"],
        #     data["points"],
        #     data["dim"],
        #     data["id"]
        # )
        # if data["use_bottom_median_as_center"]:
        #     new_obj.use_bottom_median_as_center()
        # return new_obj

    def to_pickle(self):
        self.reset_memoized()
        return self

    def set_center_ref(self, new_center_ref):
        assert new_center_ref in ['bottom_middle', 'mean']
        self._center_ref = new_center_ref
        