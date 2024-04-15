import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List
import open3d as o3d

from robot_utils.robot_data.pose_data import PoseData
from robot_utils.transform import transform_2_xytheta

from segment_track.segment import Segment
from segment_track.tracker import Tracker

from object_map_registration.object.ellipsoid import Ellipsoid
from object_map_registration.object.object import Object
from object_map_registration.register.clipper_pt_registration import ClipperPtRegistration
from object_map_registration.register.gravity_constrained_clipper_reg import GravityConstrainedClipperReg
from object_map_registration.register.dist_vol_gravity_constrained import DistVolGravityConstrained
from object_map_registration.register.dist_vol_sim_reg import DistVolSimReg
from object_map_registration.register.object_registration import InsufficientAssociationsException
from object_map_registration.utils import object_list_bounds

def centroid_from_points(segment: Segment):
    """
    Method to get a single point representing a segment.

    Args:
        segment (Segment): segment object

    Returns:
        np.array, shape=(3,): representative point
    """
    if segment.points is not None:
        pt = np.median(segment.points, axis=0)
        pt[2] = np.min(segment.points[:,2])
        return pt
    else:
        return None
    
def submap_centers_from_poses(poses: List[np.array], dist: float):
    """
    From a series of poses, generate a list of positions on the trajectory that are dist apart.

    Args:
        poses (List[np.array, shape=(3,3 or 4,4)]): list of poses
        dist (float): Distance between submap centers
    
    Returns:
        np.array, shape=(3,n)]: List of submap centers
    """
    centers = []
    for i, pose in enumerate(poses):
        if i == 0:
            centers.append(pose[:-1,-1])
        else:
            if np.linalg.norm(pose[:-1,-1] - centers[-1]) > dist:
                centers.append(pose[:-1,-1])
    np.array(centers)
    return centers

def submaps_from_maps(object_map: List[Object], centers: List[np.array], radius: float):
    """
    From a list of objects and a list of centers, generate a list of submaps centered at the centers.

    Args:
        object_map (List[Object]): List of objects
        centers (np.array, shape=(3,n)): List of submap centers
        radius (float): Radius of submaps
    
    Returns:
        List[List[Object]]: List of submaps
    """
    submaps = []
    for center in centers:
        submap = [obj for obj in object_map if np.linalg.norm(obj.centroid.flatten() - center[:obj.dim]) < radius]
        submaps.append(submap)
    return submaps
        
    
# def load_pose_data(pose_file, csv=False):
#     kmd_csv_options = {'cols': {'time': ['#timestamp_kf'],
#                                 'position': ['x', 'y', 'z'],
#                                 'orientation': ['qx', 'qy', 'qz', 'qw']},
#                     'col_nums': {'time': [0],
#                                 'position': [1,2,3],
#                                 'orientation': [5,6,7,4]},
#                     'timescale': 1e-9}
#     if not csv:
#         return PoseData(
#             data_file=os.path.expanduser(os.path.expandvars(pose_file)),
#             file_type='bag',
#             topic="/acl_jackal2/kimera_vio_ros/odometry",
#             time_tol=10.0,
#             interp=True,
#         )
#     else:
#         return PoseData(
#             data_file=os.path.expanduser(os.path.expandvars(pose_file)),
#             file_type='csv',
#             time_tol=10.0,
#             interp=True,
#             csv_options=kmd_csv_options
#         )

def main(args):
    colors = ['maroon', 'blue']
    poses = []
    trackers = []
    object_maps = [[], []]
    t0s = []
    tfs = []
    
    # extract pickled data
    for pickle_file in args.input:
        with open(os.path.expanduser(pickle_file), 'rb') as f:
            t, p = pickle.load(f)
            poses.append(p)
            trackers.append(t)

    # create objects from segments
    for i, tracker in enumerate(trackers):
        t0 = np.inf
        tf = -np.inf
        segment: Segment
        for segment in tracker.segments + tracker.segment_graveyard:
            segment.first_seen = np.min([obs.time for obs in segment.observations])
            t0 = np.min([segment.first_seen, t0])
            tf = np.max([segment.last_seen, tf])
            centroid = centroid_from_points(segment)
            if centroid is not None:

                try:
                    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                        o3d.utility.Vector3dVector(segment.points)
                    )
                except:
                    continue
                volume = obb.volume()
                # new_obj = Object(centroid.reshape(-1)[:args.dim])
                new_obj = Ellipsoid(
                    centroid.reshape(-1)[:args.dim], 
                    np.array([volume**(1/args.dim) for i in range(args.dim)]), 
                    np.eye(args.dim)
                )
                new_obj.first_seen = np.min([obs.time for obs in segment.observations])
                new_obj.last_seen = np.max([obs.time for obs in segment.observations])
                object_maps[i].append(new_obj)
        t0s.append(t0)
        tfs.append(tf)

    # create submaps
    submap_centers = [submap_centers_from_poses(pose_list, args.submap_center_dist) for pose_list in poses]
    submaps = [submaps_from_maps(object_map, center_list, args.submap_radius) for object_map, center_list in zip(object_maps, submap_centers)]

    if args.submaps_idx:
        submaps = [submaps[i][args.submaps_idx[i][0]:args.submaps_idx[i][1]] for i in range(2)]
        submap_centers = [submap_centers[i][args.submaps_idx[i][0]:args.submaps_idx[i][1]] for i in range(2)]

    # Registration setup
    clipper_angle_mat = np.empty((len(submaps[0]), len(submaps[1])))
    clipper_num_associations = np.empty((len(submaps[0]), len(submaps[1])))
    robots_nearby_mat = np.empty((len(submaps[0]), len(submaps[1])))

    # Registration method
    if args.method == 'standard':
        registration = ClipperPtRegistration(sigma=args.sigma, epsilon=args.epsilon)
    elif args.method == 'gravity':
        registration = GravityConstrainedClipperReg(sigma=args.sigma, epsilon=args.epsilon)
    elif args.method == 'distvol':
        registration = DistVolSimReg(sigma=args.sigma, epsilon=args.epsilon)
    elif args.method == 'distvolgrav':
        registration = DistVolGravityConstrained(sigma=args.sigma, epsilon=args.epsilon)
    else:
        assert False, "Invalid method"
    # registration = DistVolSimReg(sigma=args.sigma, epsilon=args.epsilon, vol_score_min=0.5, dist_score_min=0.5)

    # plot maps
    if args.show_maps:
        for i in range(2):
            fig, ax = plt.subplots()
            for obj in object_maps[i]:
                obj.plot2d(ax, color=colors[i])
            
            if args.dim == 3:
                xlim, ylim, _ = object_list_bounds(object_maps[i])
            else:
                xlim, ylim = object_list_bounds(object_maps[i])

            
            # ax.plot([position[0] for position in submap_centers[i]], [position[1] for position in submap_centers[i]], 'o', color='black')
            for center in submap_centers[i]:
                ax.add_patch(plt.Circle(center[:2], args.submap_radius, color='black', fill=False))
            ax.set_aspect('equal')
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        plt.show()
        return
    
    # iterate over pairs of submaps and create registration results
    for i in tqdm(range(len(submaps[0]))):
        for j in (range(len(submaps[1]))):
            
            submap_dist = np.linalg.norm(submap_centers[0][i] - submap_centers[1][j])
            robots_nearby_mat[i, j] = 1 if submap_dist < args.submap_radius*2 else 0

            # if show_false_positives is not set, then skip over non overlapping submaps
            if not args.show_false_positives and robots_nearby_mat[i, j] == 0:
                clipper_angle_mat[i, j] = np.nan
                clipper_num_associations[i, j] = np.nan
                continue

            if args.dim == 2:
                try:
                    T_align = registration.T_align(submaps[0][i], submaps[1][j])
                except InsufficientAssociationsException:
                    clipper_angle_mat[i, j] = 180.0
                    clipper_num_associations[i, j] = 0
                    continue
                _, _, theta = transform_2_xytheta(T_align)

            elif args.dim == 3:
                try:
                    T_align = registration.T_align(submaps[0][i], submaps[1][j])
                except InsufficientAssociationsException:
                    clipper_angle_mat[i, j] = 180.0
                    clipper_num_associations[i, j] = 0
                    continue
                theta = Rot.from_matrix(T_align[:3, :3]).magnitude()

            else:
                raise ValueError("Invalid dimension")
            
            clipper_angle_mat[i, j] = np.abs(np.rad2deg(theta))

            associations = registration.register(submaps[0][i], submaps[1][j])
            clipper_num_associations[i, j] = len(associations)

    # Create plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    mp = ax[0].imshow(robots_nearby_mat, vmin=0)
    fig.colorbar(mp)
    ax[0].set_title("Submaps Overlap")

    mp = ax[1].imshow(-clipper_angle_mat, vmax=0, vmin=-10)
    fig.colorbar(mp)
    ax[1].set_title("Registration Error (deg)")

    mp = ax[2].imshow(clipper_num_associations, vmin=0)
    fig.colorbar(mp)
    ax[2].set_title("Number of CLIPPER Associations")

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=2)
    parser.add_argument('--show-maps', action='store_true')
    parser.add_argument('--dim', '-d', type=int, default=3)
    parser.add_argument('--method', '-m', type=str, default='standard')
    parser.add_argument('--show-false-positives', '-s', action='store_true')
    parser.add_argument('--output', '-o', type=str, default=None)
    
    args = parser.parse_args()

    # constant params
    args.submap_radius = 15. # 20.0 for kimera multi data?
    args.submap_center_dist = 15.0

    args.sigma = .3
    args.epsilon = .4
    # args.submaps_idx = [[32, 42], [45, 55], ]
    # args.submaps_idx = [[38, 52], [20, 35],]
    args.submaps_idx = None
    
    main(args)