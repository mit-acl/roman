import numpy as np
import torch
import open3d as o3d
import random
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as Rot
import logging
 
# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

import matplotlib.pyplot as plt

from roman.object.object import Object
from roman.utils import get_transform_matrix

from equiv_reg.fmr_transforms import OnUnitCube 


@dataclass
class MeshObjectNoiseParams:
    """_summary_
    """
    point_stddev: float = 0.

@dataclass
class PartialViewParams:
    """Parameters for simulating partial views
    """
    num_input_points: int = 8192
    multiplier: float = 100.0
    scale_min = 0.1
    scale_max = 2.0


class MeshObject(Object):

    def __init__(self, centroid: np.array, rot_mat: np.array, mesh_canonical: o3d.geometry.TriangleMesh, dim=None):
        """A 3D object represented by a triangle mesh.
        Args:
            centroid (np.array): centroid in world frame
            rot_mat (np.array): orientation in world frame
            mesh_canonical (o3d.geometry.TriangleMesh): Mesh object in
            canonical pose without pose transformation
            dim (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(centroid, dim)
        assert self.dim == 3
        self.rot_mat = rot_mat
        # Store the mesh in canonical pose without pose transformation
        self.mesh_canonical = o3d.geometry.TriangleMesh(mesh_canonical)
        self.mesh = o3d.geometry.TriangleMesh(mesh_canonical)
        self.mesh.transform(self.get_pose())
        self._volume = None
        self._descriptor = None
        self.unitcube_op = OnUnitCube()

    def get_pose(self):
        """Get the 3d pose transformation of this object in the world frame
        """
        return get_transform_matrix(self.rot_mat, self.centroid)

    def get_mesh(self):
        return o3d.geometry.TriangleMesh(self.mesh)

    # TODO: use this in registration methods
    @property
    def center(self):
        return self.mesh.get_center().reshape(self.dim, 1)

    def transform(self, T):
        assert T.shape == (self.dim+1, self.dim+1)
        self.centroid = (T @ np.vstack([self.centroid, np.ones((1, 1))]))[:self.dim, :]
        self.rot_mat = T[:self.dim, :self.dim] @ self.rot_mat
        self.mesh.transform(T)
        return

    def add_noise(self,
                  centroid_covariance,
                  object_noise_params: MeshObjectNoiseParams = None):
        if object_noise_params is None:
            object_noise_params = MeshObjectNoiseParams()
        vertices = np.asarray(self.mesh_canonical.vertices)
        vertices += np.random.multivariate_normal(
            mean=np.zeros(3), cov=centroid_covariance).reshape(1, 3)
        vertices += np.random.normal(0, object_noise_params.point_stddev,
                                     size=vertices.shape)
        # Perturb canonical mesh
        self.mesh_canonical = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            self.mesh_canonical.triangles)
        # Generate posed mesh
        self.mesh = o3d.geometry.TriangleMesh(self.mesh_canonical)
        self.mesh.transform(self.get_pose())

    def copy(self):
        return MeshObject(self.centroid.copy(),
                          self.rot_mat.copy(),
                          self.mesh_canonical,
                          self.dim)

    def plot2d(self, ax=None, **kwargs):
        raise NotImplementedError("2D vis of 3D mesh object is not implemented.")

    def plot3d(self, ax=None, num_points=1000, z_lift=0, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        pcd = self.mesh.sample_points_uniformly(number_of_points=num_points)
        # points = np.asarray(self.mesh.vertices)
        points = np.asarray(pcd.points)
        ax.scatter(points[:, 0],
                   points[:, 1],
                   points[:, 2]+z_lift,
                   marker=".")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal', 'box')

        return ax

    def visualize(self):
        """Visualize this object using open3d

        Returns:
            _type_: _description_
        """
        o3d.visualization.draw_geometries([self.mesh])

    @property
    def volume(self):
        if self._volume is None:
            self._volume = self.estimate_volume()
        return self._volume

    def estimate_volume(self, semgenter_per_axis=10):
        """Estimate the volume by voxelizing the bounding box and checking whether sampled points 
        are inside each voxel"""
        pcd = self.mesh.sample_points_uniformly(number_of_points=semgenter_per_axis**3)
        points = np.asarray(pcd.points)
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        x_seg_size = (max_bounds[0] - min_bounds[0])/ semgenter_per_axis
        y_seg_size = (max_bounds[1] - min_bounds[1])/ semgenter_per_axis
        z_seg_size = (max_bounds[2] - min_bounds[2])/ semgenter_per_axis
        volume = 0.0
        for i in range(semgenter_per_axis):
            x = min_bounds[0] + x_seg_size * i 
            for j in range(semgenter_per_axis):
                y = min_bounds[1] + y_seg_size * j 
                for k in range(semgenter_per_axis):
                    z = min_bounds[2] + z_seg_size * k 
                    if np.any(np.bitwise_and(points < np.array([x + x_seg_size, y + y_seg_size, z + z_seg_size]), 
                                             points > np.array([x, y, z]))):
                        volume += x_seg_size * y_seg_size * z_seg_size
        return volume

    @classmethod
    def generator_fun(cls, bounds, meshes, scale_min, scale_max):
        """Generate a random mesh object.
        For now, we place the object in a location that is within a range of distance 
        from the origin.

        Args:
            bounds (_type_): Bound on distance from origin [dist_min, dist_max]
            meshes (_type_): List of objects stored as open3D meshes to sample from

        Returns:
            _type_: _description_
        """
        def gen_func():
            dist = np.random.uniform(bounds[0], bounds[1])
            theta = np.random.uniform(0., 2*np.pi)
            point = np.asarray([dist * np.cos(theta), dist * np.sin(theta), 0.])
            mesh_rand = random.choice(meshes)
            scale = np.random.uniform(scale_min, scale_max)
            return MeshObject(point,
                              Rot.from_euler('zyx', angles=[np.random.uniform(0, 180), 0, 0], degrees=True).as_matrix(),
                              mesh_rand.scale(scale, center=mesh_rand.get_center()))
        return gen_func
    
    def set_model(self, model, device, mode):
        self.model = model
        self.device = device
        self.mode = mode
        logger.debug("Setting model with mode: " + mode)

    @property
    def descriptor(self):
        # if self._descriptor is not None:
        #     return self._descriptor
        pcd = self.mesh.sample_points_uniformly(number_of_points=512)
        if self.mode == 'SE3':
            logger.debug("Getting descriptor with SE3 equivariance.")
            pcd = pcd.translate(-pcd.get_center())
            pcd, _ = self.unitcube_op(torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)))
            # pcd = torch.from_numpy(pcd)
        else:
            logger.debug("Getting descriptor with SIM3 equivariance.")
            pcd = pcd.translate(-pcd.get_center())
            pcd = np.asarray(pcd.points, dtype=np.float32)
            pcd = torch.from_numpy(pcd)

        input_pc = pcd.unsqueeze(0)
        latent = self.model.encode_inputs(input_pc.to(self.device))
        latent = latent.reshape(1, -1, 3)
        shape_descriptor = self.model.infer_shape_descriptors(latent)
        # print("Descriptor shape: ", shape_descriptor.shape)
        self._descriptor = shape_descriptor.squeeze(0).cpu().detach().numpy()
        return self._descriptor


    def simulate_partial_view(self, camera_loc: np.ndarray, params: PartialViewParams):
        """
        Simulate partial views by removing hidden points in the input object point cloud.

        Args:
            camera_loc (np.ndarray): Center of camera
            params (PartialViewParams): parameter setting

        Returns:
            torch.Tensor: Nv-by-3 visible points as torch tensor
        """
        pcd =  self.mesh.sample_points_uniformly(number_of_points=params.num_input_points)
        # Example from open3d remove hidden points
        # https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        new_mesh, pt_map = pcd.hidden_point_removal(
            camera_loc.reshape((3,1)),
            params.multiplier * diameter
        )  
        # The outputs from hidden_point_removal is in the world frame,
        # we need to transform it back to the canonical pose
        T_world_obj = self.get_pose()
        T_obj_world = np.linalg.inv(T_world_obj)
        new_mesh.transform(T_obj_world)

        return MeshObject(centroid=self.centroid, rot_mat=self.rot_mat, mesh_canonical=new_mesh)