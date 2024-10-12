import numpy as np
import matplotlib.pyplot as plt

class Object():

    def __init__(self, centroid: np.array, dim=None, id=0):
        if dim is None:
            dim = centroid.shape[0]
        assert dim == 2 or dim == 3, "dim must be 2 or 3"
        self.dim = dim
        self.centroid = centroid.reshape(self.dim, 1)
        self.id = id
        self._volume = None
        self._descriptor = None

    def transform(self, T):
        assert T.shape == (self.dim+1, self.dim+1)
        self.centroid = (T @ np.vstack([self.centroid, np.ones((1,1))]))[:self.dim,:]
        return
    
    def add_noise(self, centroid_covariance, object_noise_params = None):
        self.centroid += np.random.multivariate_normal(
            mean=np.zeros(self.dim), cov=centroid_covariance).reshape(self.dim, 1)
    
    def copy(self):
        return Object(self.centroid.copy(), self.dim)
    
    def plot2d(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.centroid[0], self.centroid[1], 'o', **kwargs)
        return ax
    
    def set_volume(self, volume):
        self._volume = volume

    def set_descriptor(self, descriptor):
        self._descriptor = descriptor
    
    @property
    def volume(self):
        if self._volume is not None:
            return self._volume
        raise NotImplementedError("Volume not implemented for generic object. Must be manually set.")
    
    @property
    def descriptor(self):
        if self._descriptor is not None:
            return self._descriptor
        raise NotImplementedError("Descriptor not implemented for generic object")
    
    @property
    def center(self):
        # Default to using object centroid.
        return self.centroid
    
    @classmethod
    def generator_fun(cls, bounds):
        return lambda: Object(np.random.uniform(bounds[:,0], bounds[:,1]))
    
    def to_pickle(self):
        return {
            "centroid": self.centroid,
            "dim": self.dim,
            "id": self.id
        }