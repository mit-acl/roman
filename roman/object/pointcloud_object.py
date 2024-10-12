import numpy as np
import open3d as o3d
# import pytorch3d.ops as ops
import torch
from scipy.spatial.transform import Rotation as Rot
from numpy.linalg import svd
from scipy.stats import entropy
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import cached_property

# from equiv_reg.transforms import apply_rot, gen_randrot
# from equiv_reg.fmr_transforms import OnUnitCube

from roman.object.object import Object
from roman.utils import get_transform_matrix


@dataclass
class PointCloudObjectNoiseParams:
    """_summary_
    """
    point_stddev: float = 0.


@dataclass
class PartialViewParams:
    """Parameters for simulating partial views
    """
    num_output_points: int = 512
    camera_dist_min: float = 2.0
    camera_dist_max: float = 3.0
    camera_min_z: float = -0.5
    camera_max_z: float = 0.0
    multiplier: float = 100.0
    debug: bool = False
    erosion: bool = False
    erosion_radius_min: float = 0.0
    erosion_radius_max: float = 0.0

def cam_random_position_generator(partial_view_params: PartialViewParams):
    def gen_fun():
        height = np.random.uniform(low=partial_view_params.camera_min_z, high=partial_view_params.camera_max_z)
        camera_dist = np.random.uniform(low=partial_view_params.camera_dist_min, high=partial_view_params.camera_dist_max)
        camera_loc = torch.tensor([camera_dist, 0.0, height])
        if np.random.rand() > 0.8:
            Rz, _ = gen_randrot(mag_max=180, mag_random=False, rot_axis=np.asarray([0, 0, 1], dtype=np.float32))
        else:
            Rz, _ = gen_randrot(mag_max=180, mag_random=True, rot_axis=np.asarray([0, 0, 1], dtype=np.float32))
        camera_loc = apply_rot(Rz, camera_loc)
        return camera_loc.squeeze().numpy()

    return gen_fun



class PointCloudObject(Object):

    def __init__(self, centroid: np.array, rot_mat: np.array, points: np.ndarray, dim=None, id=0):
        """A 3D object represented by a point cloud.
        Args:
            centroid (np.array): centroid in world frame
            rot_mat (np.array): orientation in world frame
            points (np.array): n-by-3 3D points in
            canonical pose without pose transformation
            dim (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(centroid, dim, id)
        assert self.dim == 3
        assert points.ndim == 2 and points.shape[1] == 3
        self.rot_mat = rot_mat
        # Store the point cloud in canonical pose without pose transformation
        self.pcd_canonical = o3d.geometry.PointCloud()
        self.pcd_canonical.points = o3d.utility.Vector3dVector(points)
        self.pcd = o3d.geometry.PointCloud(self.pcd_canonical)
        self.pcd.transform(self.get_pose())
        # self.unitcube_op = OnUnitCube()
        self._volume = None
        self._max_extent = None
        self._min_extent = None
        self._descriptor = None
        self._use_bottom_median_as_center = False
        self._predicted_volume = None
        self._predicted_full_pcd = None

    def copy(self):
        cp = PointCloudObject(
            self.centroid.copy(),
            self.rot_mat.copy(),
            np.asarray(self.pcd_canonical.points),
            self.dim,
            self._id
        )
        cp._volume = self._volume
        cp._max_extent = self._max_extent
        cp._min_extent = self._min_extent
        if self._use_bottom_median_as_center:
            cp.use_bottom_median_as_center()
        return cp

    @property
    def center(self):
        if self._use_bottom_median_as_center:
            pt = np.median(self.get_points(), axis=0)
            pt[2] = np.min(self.get_points()[:,2])
            return pt
        return self.pcd.get_center().reshape(self.dim, 1)
    
    def set_model(self, model, device, mode, descriptor_model=None):
        self.model = model
        self.device = device
        self.mode = mode
        self.descriptor_model = descriptor_model
        # print("Setting model with mode: " + mode)

    @property
    def descriptor(self):
        # if self._descriptor is not None:
        #     return self._descriptor
        pcd =  torch.from_numpy(np.asarray(self.pcd.points)).unsqueeze(dim=0)
        pcd_subsampled, _ = ops.sample_farthest_points(pcd, K=512)
        pcd_subsampled = pcd_subsampled.squeeze().cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_subsampled)

        if self.mode == 'SE3':
            # print("Getting descriptor with SE3 equivariance.")
            pcd = pcd.translate(-pcd.get_center())
            pcd, _ = self.unitcube_op(torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)))
        else:
            # print("Getting descriptor with SIM3 equivariance.")
            pcd = pcd.translate(-pcd.get_center())
            pcd = np.asarray(pcd.points, dtype=np.float32)
            pcd = torch.from_numpy(pcd)

        input_pc = pcd.unsqueeze(0)
        latent = self.model.encode_inputs(input_pc.to(self.device))
        # self._latent_pcd = latent
        latent = latent.reshape(1, -1, 3)
        # self._latent = latent
        shape_descriptor = self.model.infer_shape_descriptors(latent)
        # if self.descriptor_model is not None:
        #     contrastive_descriptor = self.descriptor_model(torch.nn.functional.normalize(shape_descriptor)).squeeze(0).cpu().detach().numpy()
        #     shape_descriptor = shape_descriptor.squeeze(0).cpu().detach().numpy()
        #     self._descriptor = np.concatenate((shape_descriptor, contrastive_descriptor))
        #     # print("Descriptor shape: ", shape_descriptor.shape)
        #     # print("Contrastive Descriptor shape: ", contrastive_descriptor.shape)
        #     # print("Concatenated descriptor shape: ", self._descriptor.shape)
        #     return self._descriptor
        
        self._descriptor = shape_descriptor.squeeze(0).cpu().detach().numpy()
        return self._descriptor

    def partial_pcd_completion(self, points_iou, THRESHOLD = 0.5):
        inputs = torch.from_numpy(np.asarray(self.pcd.points, dtype=np.float32)).to(self.device)
        # print(inputs.shape)
        inputs_unitcube, (scale, center) = self.unitcube_op(inputs)
        inputs_unitcube = inputs_unitcube.unsqueeze(dim=0)
        points_iou = points_iou.to(self.device)
        # print("Points iou shape: ", points_iou.shape)
        # print("Inputs shape: ", inputs_unitcube.shape)
        with torch.no_grad():
            p_out = self.model(points_iou, inputs_unitcube, sample=False)
        
        occ_iou_hat_np = (p_out.probs >= THRESHOLD).cpu().numpy()
        indices_pred = np.nonzero(occ_iou_hat_np)[1]

        points_ = points_iou.to('cpu').numpy()
        points_pred = points_[0, indices_pred, :]

        # print(f"Shape of predicted occupied points on unit cube: {points_pred.shape}")
        # print("Scale: ", scale)
        # print("Center: ", center)
        self._predicted_full_pcd = scale*( torch.from_numpy(points_pred).to(self.device ) + center)
        # print(f"Shape of predicted occupied points: {self._predicted_full_pcd.shape}")
    

        return self._predicted_full_pcd

    def predict_volume(self):
        if self._predicted_full_pcd is None:
            self.partial_pcd_completion()
            
        if len(self._predicted_full_pcd) < 4:
            self._predicted_volume = 0.0
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self._predicted_full_pcd.to('cpu').numpy())
            pcd += self.pcd
            try:
                self._predicted_volume = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points).volume()
            except:
                self._predicted_volume = 0.0

        return self._predicted_volume
    
    @property
    def predicted_volume(self):
        if self._predicted_volume is None:
            self.predict_volume()
        return self._predicted_volume
    
    def simulate_partial_view(self, camera_loc: np.ndarray,
                    params: PartialViewParams,
                    scale=1.0) -> torch.Tensor:
        """Simulate partial views by removing hidden points in the input object point cloud.

        Args:
            object_points (torch.tensor): Complete object as torch N-by-3 tensor
            camera_loc (np.ndarray): Center of camera
            params (PartialViewParams): parameter setting

        Returns:
            torch.Tensor: Nv-by-3 visible points as torch tensor
        """
        pcd = self.pcd
        # Example from open3d remove hidden points
        # https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        mesh, pt_map = pcd.hidden_point_removal(
            camera_loc, 
            params.multiplier * diameter
        )  
        num_out = params.num_output_points
        
        # Make output contains the specified number of points
        
        # Implementation 1: sample from mesh
        # pcd_out = mesh.sample_points_uniformly(number_of_points=num_out)
        # points_out = np.asarray(pcd_out.points, dtype=np.float32)

        # Implementation 2: sampling from points (possibly contain duplicates)
        pcd = pcd.select_by_index(pt_map)
        visible_points = np.asarray(pcd.points, dtype=np.float32)

        num_visible = visible_points.shape[0]
        if num_visible >= params.num_output_points:
            indices = torch.randperm(num_visible)[:num_out]
            # logger.debug("[partial_view] Sample {} out of {} visible points".format(num_out, num_visible))
        else:
            indices = np.random.randint(num_visible, size=num_out)
            # logger.debug("[partial_view] Not enough output points: have {} but want {}".format(num_visible, num_out))
        points_out = visible_points[indices, :]

        pcd_partial = o3d.geometry.PointCloud()
        pcd_partial.points = o3d.utility.Vector3dVector(points_out)
        # print("Generate partial views of point clouds.")
        return PointCloudObject(pcd_partial.get_center(), self.rot_mat, points_out)

    
    @classmethod
    def generator_fun(cls, bounds, pcds, scale_min, scale_max):
        """Generate a random point cloud object.
        For now, we place the object in a location that is within a range of distance 
        from the origin.

        Args:
            bounds (_type_): Bound on distance from origin [dist_min, dist_max]
            pcds (_type_): List of objects stored as open3D pcd to sample from

        Returns:
            _type_: _description_
        """
        def gen_func():
            dist = np.random.uniform(bounds[0], bounds[1])
            theta = np.random.uniform(0., 2*np.pi)
            # z_rand = np.random.uniform(0., 2.0)
            point = np.asarray([dist * np.cos(theta), dist * np.sin(theta), 0.0])
            pcd = random.choice(pcds)
            pcd = pcd.translate(point - pcd.get_center())
            scale = np.random.uniform(scale_min, scale_max)
            pcd.scale(scale, center=pcd.get_center())

            return PointCloudObject(pcd.get_center(),
                              Rot.from_euler('zyx', angles=[np.random.uniform(0, 360), 0, 0], degrees=True).as_matrix(),
                              np.asarray(pcd.points))
        return gen_func
    
    def get_pose(self):
        """Get the 3d pose transformation of this object in the world frame
        """
        return get_transform_matrix(self.rot_mat, self.centroid)

    def get_points(self):
        """Get the 3D points in this object as n-by-3 np matrix 
        """
        return np.asarray(self.pcd.points)

    def transform(self, T):
        assert T.shape == (self.dim+1, self.dim+1)
        self.centroid = (T @ np.vstack([self.centroid, np.ones((1, 1))]))[:self.dim, :]
        self.rot_mat = T[:self.dim, :self.dim] @ self.rot_mat
        self.pcd.transform(T)
        return

    def add_noise(self, centroid_covariance, object_noise_params = None):
        return super().add_noise(centroid_covariance, object_noise_params)

    def plot2d(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.center[0], self.center[1], 'o', **kwargs)
        return ax

    def plot3d(self, ax=None, z_lift=0, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        points = self.get_points()
        ax.scatter(points[:, 0],
                   points[:, 1],
                   points[:, 2]+z_lift,
                   marker=".")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal', 'box')

        return ax
    
    @property
    def volume(self):
        if self._volume is None:
            if len(self.get_points()) < 4:
                self._volume = 0.0
            else:
                try:
                    self._volume = o3d.geometry.OrientedBoundingBox.create_from_points(self.pcd.points).volume()
                except: # can fail if points are colinear/coplanar
                    self._volume = 0.0
                # self._volume = np.max(o3d.geometry.OrientedBoundingBox.create_from_points(self.pcd.points).extent)
            # self._volume = self.estimate_volume()
        return self._volume
    
    @cached_property
    def extent(self):
        return o3d.geometry.OrientedBoundingBox.create_from_points(self.pcd.points).extent
    
    @property
    def max_extent(self):
        if self._max_extent is None:
            if len(self.get_points()) < 4:
                self._max_extent = 0.0
            else:
                self._max_extent = np.max(o3d.geometry.OrientedBoundingBox.create_from_points(self.pcd.points).extent)
        return self._max_extent
    
    @property
    def min_extent(self):
        if self._min_extent is None:
            if len(self.get_points()) < 4:
                self._min_extent = 0.0
            else:
                self._min_extent = np.min(o3d.geometry.OrientedBoundingBox.create_from_points(self.pcd.points).extent)
        return self._min_extent

    def estimate_volume(self, semgenter_per_axis=10):
        """Estimate the volume by voxelizing the bounding box and checking whether sampled points 
        are inside each voxel"""
        points = self.get_points()
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


    def use_bottom_median_as_center(self):
        """Use the median of the points with the smallest z value as the center
        """
        self._use_bottom_median_as_center = True

    def normalized_eigenvalues(self):
      """Compute the normalized eigenvalues of the covariance matrix
      as a np array [e1, e2, e3]
      e1 >= e2 >= e3 so that the sum is one
      """
      _, C = self.pcd.compute_mean_and_covariance()
      _, eigvals, _ = svd(C)  # svd return in descending order
      return eigvals / eigvals.sum()

    def linearity(self, e: np.ndarray=None):
      """ Large if similar to a 1D line (Weinmann et al. ISPRS 2014)

      Args:
          e (np.ndarray): normalized eigenvalues of this point cloud
      """
      if e is None:
        e = self.normalized_eigenvalues()
      return (e[0]-e[1]) / e[0]

    def planarity(self, e: np.ndarray=None):
      """ Large if similar to a 2D plane (Weinmann et al. ISPRS 2014)
      Args:
          e (np.ndarray): normalized eigenvalues of this point cloud
      """
      if e is None:
        e = self.normalized_eigenvalues()
      return (e[1]-e[2]) / e[0]

    def scattering(self, e: np.ndarray=None):
      """Large if this object is 3D, i.e., neither a line nor a plane (Weinmann et al. ISPRS 2014)

      Args:
          e (np.ndarray): normalized eigenvalues of this point cloud
      """
      if e is None:
        e = self.normalized_eigenvalues()
      return e[2] / e[0]

    def omnivariance(self, e: np.ndarray=None):
      """ Cubic root of determinant (Weinmann et al. ISPRS 2014)

      Args:
          e (np.ndarray): normalized eigenvalues of this point cloud
      """
      if e is None:
        e = self.normalized_eigenvalues()
      return np.cbrt(e.prod())
    
    def anisotropy(self, e: np.ndarray=None):
      """ Relative gap between dominant eigenvalues (Weinmann et al. ISPRS 2014)

      Args:
          e (np.ndarray): normalized eigenvalues of this point cloud
      """
      if e is None:
        e = self.normalized_eigenvalues()
      return (e[0]-e[2]) / e[0]

    def eigenentropy(self, e: np.ndarray=None):
      """Entropy measure of eigenvalues (Weinmann et al. ISPRS 2014)

      Args:
          e (np.ndarray): normalized eigenvalues of this point cloud
      """
      if e is None:
        e = self.normalized_eigenvalues()
      return entropy(e)

    def change_of_curvature(self, e: np.ndarray=None):
      """ 'surface variance' (Weinmann et al. ISPRS 2014)

      Args:
          e (np.ndarray): normalized eigenvalues of this point cloud
      """
      if e is None:
        e = self.normalized_eigenvalues()
      return e[2]
        
    def to_pickle(self):
        return {
            "centroid": self.centroid,
            "rot_mat": self.rot_mat,
            "points": np.asarray(self.pcd_canonical.points),
            "dim": self.dim,
            "use_bottom_median_as_center": self._use_bottom_median_as_center,
            "id": self._id,
        }
        
    @classmethod
    def from_pickle(cls, data):
        new_obj = cls(
            data["centroid"],
            data["rot_mat"],
            data["points"],
            data["dim"],
            data["id"]
        )
        if data["use_bottom_median_as_center"]:
            new_obj.use_bottom_median_as_center()
        return new_obj

    @classmethod
    def from_pcd(cls, pcd_file_path):
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        new_obj = cls(
            np.asarray(pcd.get_center()),
            np.eye(3),
            np.asarray(pcd.points),
            3
        )
        return new_obj
    