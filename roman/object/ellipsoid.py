import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from roman.object.object import Object

@dataclass
class EllipsoidNoiseParams:

    euler_angles_covariance: np.array
    axes_covariance: np.array
    min_axis_len: float = 0.1

class Ellipsoid(Object):

    def __init__(self, centroid: np.array, axes: np.array, rot_mat: np.array, dim=None):
        """
        Ellipsoid object.

        Args:
            centroid (np.array, shape=(d,1)): Centroid of the ellipsoid.
            axes (np.array, shape=(d)): Ellipsoid axes lengths.
            rot_mat (np.array, shape=(d,d)): Rotation matrix.
            dim (int, optional): Dimension of ellipsoid (d, should be 2 or 3). Defaults to None.
        """
        super().__init__(centroid, dim)
        self.axes = axes.reshape(-1)
        self.rot_mat = rot_mat

    def transform(self, T):
        assert T.shape == (self.dim+1, self.dim+1)
        self.centroid = (T @ np.vstack([self.centroid, np.ones((1,1))]))[:self.dim,:]
        self.rot_mat = T[:self.dim,:self.dim] @ self.rot_mat
        return
    
    def add_noise(self, centroid_covariance, object_noise_params: EllipsoidNoiseParams):
        self.centroid += np.random.multivariate_normal(
            mean=np.zeros(self.dim), cov=centroid_covariance).reshape(self.dim, 1)
        if self.dim == 3:
            euler_angles_noise = np.random.multivariate_normal(
                mean=np.zeros(self.dim), cov=object_noise_params.euler_angles_covariance).reshape(-1)
            rot_noise = Rot.from_euler('xyz', euler_angles_noise).as_matrix()
        elif self.dim == 2:
            rot_noise = Rot.from_euler('z', np.random.normal(0, object_noise_params.euler_angles_covariance)).as_matrix()[:2,:2]
        self.rot_mat = rot_noise @ self.rot_mat
        self.axes += np.random.multivariate_normal(
            mean=np.zeros(self.dim), cov=object_noise_params.axes_covariance).reshape(-1)
        self.axes = np.maximum(self.axes, object_noise_params.min_axis_len)
    
    def copy(self):
        return Ellipsoid(self.centroid.copy(), self.axes.copy(), self.rot_mat.copy(), self.dim)
    
    def plot2d(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        # ax.plot(self.centroid[0], self.centroid[1], 'o', **kwargs)
        artist = Ellipse(xy=self.centroid.reshape(-1)[:2], width=self.axes[0], height=self.axes[1], 
                             angle=np.arctan2(self.rot_mat[1,0], self.rot_mat[0,0])*180/np.pi, facecolor='none', **kwargs)
        ax.add_artist(artist)
        return ax
    
    @property
    def volume(self):
        if self.dim == 2:
            return np.pi * self.axes[0] * self.axes[1]
        elif self.dim == 3:
            return 4/3 * np.pi * self.axes[0] * self.axes[1] * self.axes[2]
        
    @property
    def extent(self):
        return np.sort(self.axes.reshape(-1))
    
    @classmethod
    def generator_fun(cls, bounds, axes_bounds):
        assert bounds.shape == axes_bounds.shape
        dim = bounds.shape[0]

        if dim == 2:
            return lambda: Ellipsoid(np.random.uniform(bounds[:,0], bounds[:,1]), 
                                     np.random.uniform(axes_bounds[:,0], axes_bounds[:,1]), 
                                     Rot.from_euler('z', np.random.uniform(0, 2*np.pi)).as_matrix()[:2,:2])
        elif dim == 3:
            return lambda: Ellipsoid(np.random.uniform(bounds[:,0], bounds[:,1]), 
                                    np.random.uniform(axes_bounds[:,0], axes_bounds[:,1]), 
                                    Rot.random().as_matrix())
        else:
            raise Exception("Ellipsoid only implemented for 2D and 3D")
    
