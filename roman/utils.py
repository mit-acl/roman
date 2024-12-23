###########################################################
#
# utils.py
#
# ROMAN package utility functions
#
# Authors: Mason Peterson
#
# Dec. 23, 2024
#
###########################################################

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as Rot
from os.path import expandvars, expanduser

from roman.object.object import Object

def plot_correspondences(map1: List[Object], map2: List[Object], correspondences: np.array, ax=None,
                         map1_kwargs={'color':'maroon'}, map2_kwargs={'color':'blue'}, correspondence_kwargs={'color':'lawngreen'}):
    if ax is None:
        fig, ax = plt.subplots()

    for obj in map1:
        obj.plot2d(ax=ax, **map1_kwargs)

    for obj in map2:
        obj.plot2d(ax=ax, **map2_kwargs)

    if not (type(correspondences) == list and len(correspondences[0].shape) == 2):
        correspondences = [correspondences]
        correspondence_kwargs = [correspondence_kwargs]
    else:
        assert len(correspondences) == len(correspondence_kwargs)
        assert type(correspondence_kwargs) == list

    for c_set, c_kwargs in zip(correspondences, correspondence_kwargs):
        x = []
        y = []
        for i in range(c_set.shape[0]):
            x += [map1[c_set[i,0]].centroid.item(0), map2[c_set[i,1]].centroid.item(0), np.nan]
            y += [map1[c_set[i,0]].centroid.item(1), map2[c_set[i,1]].centroid.item(1), np.nan]
        ax.plot(x, y, **c_kwargs)
        
    ax.set_aspect('equal')

    return ax

def plot_correspondences_3d(map1: List[Object], map2: List[Object], correspondences: np.array, ax=None, z_lift=3,
                         map1_kwargs={'color':'maroon'}, map2_kwargs={'color':'blue'}, correspondence_kwargs={'color':'lawngreen'}):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    for obj in map1:
        obj.plot3d(ax=ax, **map1_kwargs)

    for obj in map2:
        obj.plot3d(ax=ax, z_lift=z_lift, **map2_kwargs)

    for i in range(correspondences.shape[0]):
        ax.plot([map1[correspondences[i,0]].centroid[0], map2[correspondences[i,1]].centroid[0]], 
                 [map1[correspondences[i,0]].centroid[1], map2[correspondences[i,1]].centroid[1]],
                 [map1[correspondences[i,0]].centroid[2], map2[correspondences[i,1]].centroid[2]+z_lift],
                 **correspondence_kwargs)
        
    ax.set_aspect('equal')

    return ax

def plot_correspondences_pcd(map1: List[Object], map2: List[Object], correspondences: np.array, 
                             cam1=None, cam2=None,
                             ax=None, z_lift=3,
                         map1_kwargs={'color':'maroon'}, map2_kwargs={'color':'blue'}, correspondence_kwargs={'color':'lawngreen'}):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    for obj in map1:
        obj.plot3d(ax=ax, **map1_kwargs)

    for obj in map2:
        obj.plot3d(ax=ax, z_lift=z_lift, **map2_kwargs)

    for i in range(correspondences.shape[0]):
        center1=map1[correspondences[i,0]].center
        center2=map2[correspondences[i,1]].center
        ax.plot([center1[0], center2[0]], 
                 [center1[1], center2[1]],
                 [center1[2], center2[2]+z_lift],
                 **correspondence_kwargs)

    ax.set_aspect('equal')
    

    return ax


def transform_vec(T, vec):
    unshaped_vec = vec.reshape(-1)
    resized_vec = np.concatenate(
        [unshaped_vec, np.zeros((T.shape[0] - 1 - unshaped_vec.shape[0]))]).reshape(-1)
    resized_vec = np.concatenate(
        [resized_vec, np.ones((T.shape[0] - resized_vec.shape[0]))]).reshape((-1, 1))
    transformed = T @ resized_vec
    return transformed.reshape(-1)[:unshaped_vec.shape[0]].reshape(vec.shape) 

def transform(T, vecs, axis=0):
    if len(vecs.reshape(-1)) == 2 or len(vecs.reshape(-1)) == 3:
        return transform_vec(T, vecs)
    vecs_horz_stacked = vecs if axis==1 else vecs.T
    zero_padded_vecs = np.vstack(
        [vecs_horz_stacked, np.zeros((T.shape[0] - 1 - vecs_horz_stacked.shape[0], vecs_horz_stacked.shape[1]))]
    )
    one_padded_vecs = np.vstack(
        [zero_padded_vecs, np.ones((1, vecs_horz_stacked.shape[1]))]
    )
    transformed = T @ one_padded_vecs
    transformed = transformed[:vecs_horz_stacked.shape[0],:] 
    return transformed if axis == 1 else transformed.T

def get_transform_matrix(R, t):
    """Assemble SE(d) transformation matrix 
    as a (d+1)-by-(d+1) matrix
    from SO(d) and R^d.

    Args:
        R (_type_): rotation matrix
        t (_type_): translation vecor
    """
    d = R.shape[0]
    assert R.shape == (d, d)
    assert t.shape == (d, 1)
    T = np.eye(d+1)
    T[0:d, 0:d] = R
    T[:d, d] = t.flatten()
    return T

def object_list_bounds(obj_list: List[Object]):
    centroids = np.array([obj.center.reshape(-1) for obj in obj_list])
    # dims = np.array([obj.dim for obj in obj_list])
    # assert all(i == dims[0] for i in dims)
    # dim = dims[0]

    return np.hstack([np.min(centroids, axis=0).reshape((-1,1)), np.max(centroids, axis=0).reshape((-1,1))])

def rotation_rm_roll_pitch(R):
    return Rot.from_euler('z', Rot.from_matrix(R).as_euler('ZYX')[0]).as_matrix()

def transform_rm_roll_pitch(T):
    T[:3,:3] = Rot.from_euler('z', Rot.from_matrix(T[:3,:3]).as_euler('ZYX')[0]).as_matrix()
    return T

def expandvars_recursive(path):
    """Recursively expands environment variables in the given path."""
    while True:
        expanded_path = expandvars(path)
        if expanded_path == path:
            return expanduser(expanded_path)
        path = expanded_path