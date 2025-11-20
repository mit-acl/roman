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
import torch
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

def combinedicts_recursive(d1, d2):
    """
    Combine d1 and d2:

    - if d1[k] and d2[k] are both dicts, combine them recursively
    - otherwise:
        - use d2[k] if k is in d2
        - use d1[k] if k is not in d2
    """
    res = {}
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            res[k] = combinedicts_recursive(d1[k], v)
        else:
            res[k] = v
    for k, v in d1.items():
        if k not in d2:
            res[k] = v
    return res

def aabb_intersects(p1, p2):
    """Check if the axis-aligned bounding boxes of two pointclouds intersect."""
    p1_min = np.min(p1, axis=0)
    p1_max = np.max(p1, axis=0)
    p2_min = np.min(p2, axis=0)
    p2_max = np.max(p2, axis=0)

    return (p1_min[0] <= p2_max[0] and p1_max[0] >= p2_min[0] and
            p1_min[1] <= p2_max[1] and p1_max[1] >= p2_min[1] and
            p1_min[2] <= p2_max[2] and p1_max[2] >= p2_min[2])