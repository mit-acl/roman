import numpy as np
from geomloss import SamplesLoss
import random
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import shapely.geometry as shapely

from robotdatapy.transform import transform
from roman.object.object import Object

class Polytope(Object):

    def __init__(self, vertices, hull=None):
        self.vertices = vertices
        if hull is None:
            hull = ConvexHull(self.vertices)
        self.hull = hull
        self.dim = vertices.shape[1]

    def transform(self, T):
        self.vertices = transform(T, self.vertices)
        self.hull = ConvexHull(self.vertices)

    def add_noise(self, centroid_covariance, std_per_dim):
        self.vertices += np.random.normal(loc=0.0, scale=std_per_dim, size=self.vertices.shape)
        self.vertices += np.random.multivariate_normal(
            mean=np.zeros(self.dim), cov=centroid_covariance).reshape(self.dim, 1)
        self.hull = ConvexHull(self.vertices)
        
    def copy(self):
        return Polytope(self.vertices.copy())

    def plot2d(self, ax: plt.Axes = None, color='b'):

        if ax is None:
            fig, ax = plt.subplots()
        # ax.plot(self.vertices[self.hull.vertices,0], self.vertices[self.hull.vertices,1], 'o')
        for simplex in self.hull.simplices:
            plt.plot(self.vertices[simplex, 0], self.vertices[simplex, 1], linestyle='-', color=color)
        
        return ax

    def center_at_origin(self):
        self.vertices -= self.centroid
        self.hull = ConvexHull(self.vertices)
        
    def iou(self, other):
        p1 = shapely.Polygon(self.vertices)
        p2 = shapely.Polygon(other.vertices)
        return p1.intersection(p2).area / p1.union(p2).area

    @property
    def centroid(self):
        centroid = shapely.Polygon(self.vertices).centroid
        return np.array([centroid.x, centroid.y])
    
    @property
    def volume(self):
        if self.dim == 2:
            return self.hull.area
        elif self.dim == 3:
            return self.hull.volume
        else:
            raise ValueError('Volume not implemented for dim > 3')


    @classmethod
    def sample_polytope_nd(cls, offset, size, dim):
        '''
        Sample an nd polytope, which number of vertices ranging from dim + 1 to 3*(dim + 1) (this is somewhat arbitrary).
        '''
        return lambda : Polytope(
            np.array([np.array([random.uniform(-size, size) for i in range(dim)]) \
                      for _ in range(random.randint(dim + 1, 3*(dim + 1)))]) + np.array(offset))
'''
Credit: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
'''
def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0