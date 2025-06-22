import numpy as np
from dataclasses import dataclass
import open3d as o3d
import functools

@dataclass(frozen=True)
class VoxelGrid():
    """
    Voxel bounding box data class
    """
    min_corner: np.ndarray
    max_corner: np.ndarray
    voxel_size: float
    voxels: np.ndarray
    
    @functools.cached_property
    def num_occupied(self):
        return np.sum(self.voxels)
    
    @functools.cached_property
    def volume(self):
        return self.num_occupied * self.voxel_size**3

    @property
    def min_corner_int(self):
        return (self.min_corner / self.voxel_size).astype(np.int64)
    
    @property
    def max_corner_int(self):
        return (self.max_corner / self.voxel_size).astype(np.int64)
    
    def intersection(self, other):
        assert type(other) == VoxelGrid, "Can only intersect with another VoxelGrid"
        assert self.voxel_size == other.voxel_size, "Voxel sizes must be the same"
        min_corner = np.maximum(self.min_corner, other.min_corner)
        max_corner = np.minimum(self.max_corner, other.max_corner)
        
        # voxel grids do not overlap
        if np.any(min_corner >= max_corner):
            return 0.0
        
        # downsize grids to intersection
        # start by converting coordinates to integer indices
        min_corner_int = np.maximum(self.min_corner_int, other.min_corner_int)
        max_corner_int = np.minimum(self.max_corner_int, other.max_corner_int)
        
        # print(self.voxels.shape)
        # print(min_corner_int)
        # print(max_corner_int)
        # print(self.min_corner_int)
        # print(self.max_corner_int)
        self_sub_grid = self.voxels[
            min_corner_int.item(0) - self.min_corner_int.item(0):max_corner_int.item(0) - self.min_corner_int.item(0), 
            min_corner_int.item(1) - self.min_corner_int.item(1):max_corner_int.item(1) - self.min_corner_int.item(1), 
            min_corner_int.item(2) - self.min_corner_int.item(2):max_corner_int.item(2) - self.min_corner_int.item(2)
        ]
        
        other_sub_grid = other.voxels[
            min_corner_int.item(0) - other.min_corner_int.item(0):max_corner_int.item(0) - other.min_corner_int.item(0), 
            min_corner_int.item(1) - other.min_corner_int.item(1):max_corner_int.item(1) - other.min_corner_int.item(1), 
            min_corner_int.item(2) - other.min_corner_int.item(2):max_corner_int.item(2) - other.min_corner_int.item(2)
        ]
        
        assert self_sub_grid.shape == other_sub_grid.shape, "Subgrids do not have the same shape"
        intersection_grid = self_sub_grid * other_sub_grid
        num_occupied = np.sum(intersection_grid)
        # intersection_grid /= intersection_grid
        # num_occupied = np.nansum(intersection_grid)
        return num_occupied * self.voxel_size**3
        
    def union(self, other):
        assert type(other) == VoxelGrid, "Can only union with another VoxelGrid"
        return self.volume + other.volume - self.intersection(other)
    
    def iou(self, other, iom_as_iou=False):
        assert type(other) == VoxelGrid, "Can only calculate IoU with another VoxelGrid"
        intersection = self.intersection(other)
        if iom_as_iou: 
            if np.minimum(self.volume, other.volume) == 0: return 0.0
            return intersection / np.minimum(self.volume, other.volume)
        else: 
            if (self.volume + other.volume - intersection) == 0: return 0.0
            return intersection / (self.volume + other.volume - intersection)
    
    @classmethod
    def from_points(cls, points: np.ndarray, voxel_size: float):
        """
        Create a voxel grid from a point cloud
        """
        min_corner = np.array([np.floor(np.min(points, axis=0) / voxel_size) * voxel_size])
        max_corner = np.array([np.ceil(np.max(points, axis=0) / voxel_size) * voxel_size])
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        voxels_o3d = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size=voxel_size)
        voxel_o3d_list = voxels_o3d.get_voxels()  # returns list of voxels
        indices = np.stack(list(vx.grid_index for vx in voxel_o3d_list))
        voxels = np.zeros(((max_corner - min_corner) / voxel_size + 2).astype(np.uint32).reshape(-1), dtype=np.uint8)
        # print(((max_corner - min_corner) / voxel_size).astype(np.uint32))
        # print(((max_corner - min_corner) / voxel_size).astype(np.uint32).reshape(-1))
        # print(voxels.shape)
        # print(indices)
        voxels[indices[:,0], indices[:,1], indices[:,2]] = 1
        return cls(min_corner, max_corner, voxel_size, voxels)