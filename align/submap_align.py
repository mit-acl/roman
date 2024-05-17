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
from robot_utils.geometry import circle_intersection

from segment_track.segment import Segment
from segment_track.tracker import Tracker

from object_map_registration.object.ellipsoid import Ellipsoid
from object_map_registration.object.object import Object
from object_map_registration.object.pointcloud_object import PointCloudObject
from object_map_registration.register.clipper_pt_registration import ClipperPtRegistration
from object_map_registration.register.gravity_constrained_clipper_reg import GravityConstrainedClipperReg
from object_map_registration.register.dist_vol_gravity_constrained import DistVolGravityConstrained
from object_map_registration.register.dist_vol_sim_reg import DistVolSimReg
from object_map_registration.register.object_registration import InsufficientAssociationsException
from object_map_registration.utils import object_list_bounds
    
    
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
        submap = [obj for obj in object_map if np.linalg.norm(obj.center.flatten()[:2] - center[:2]) < radius]
        submaps.append(submap)
    return submaps

def create_submaps(pickle_files: List[str], submap_radius: float, submap_center_dist: float, show_maps=False):
    colors = ['maroon', 'blue']
    poses = []
    trackers = []
    object_maps = [[], []]
    
    # extract pickled data
    for pickle_file in pickle_files:
        with open(os.path.expanduser(pickle_file), 'rb') as f:
            t, p = pickle.load(f)
            poses.append(p)
            trackers.append(t)

    # create objects from segments
    for i, tracker in enumerate(trackers):
        segment: Segment
        for segment in tracker.segments + tracker.segment_graveyard:
            if segment.points is not None and len(segment.points) > 0:
                # new_obj = PointCloudObject(np.mean(segment.points, axis=0), np.eye(3), segment.points - np.mean(segment.points, axis=0), dim=3)
                new_obj = PointCloudObject(np.zeros(3), np.eye(3), segment.points, dim=3)
                new_obj.use_bottom_median_as_center()
                object_maps[i].append(new_obj)

    # create submaps
    submap_centers = [submap_centers_from_poses(pose_list, submap_center_dist) for pose_list in poses]
    submaps = [submaps_from_maps(object_map, center_list, submap_radius) for object_map, center_list in zip(object_maps, submap_centers)]
    
    # plot maps
    if show_maps:
        for i in range(2):
            fig, ax = plt.subplots()
            for obj in object_maps[i]:
                obj.plot2d(ax, color=colors[i])
            
            bounds = object_list_bounds(object_maps[i])
            if len(bounds) == 3:
                xlim, ylim, _ = bounds
            else:
                xlim, ylim = bounds

            
            # ax.plot([position[0] for position in submap_centers[i]], [position[1] for position in submap_centers[i]], 'o', color='black')
            for center in submap_centers[i]:
                ax.add_patch(plt.Circle(center[:2], submap_radius, color='black', fill=False))
            ax.set_aspect('equal')
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        plt.show()
        return
    return submap_centers, submaps
        

def main(args):
    submap_centers, submaps = create_submaps(args.input, args.submap_radius, args.submap_center_dist, show_maps=args.show_maps)

    if args.submaps_idx:
        submaps = [submaps[i][args.submaps_idx[i][0]:args.submaps_idx[i][1]] for i in range(2)]
        submap_centers = [submap_centers[i][args.submaps_idx[i][0]:args.submaps_idx[i][1]] for i in range(2)]

    # Registration setup
    clipper_angle_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    clipper_dist_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    clipper_num_associations = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    robots_nearby_mat = np.empty((len(submaps[0]), len(submaps[1])))

    # Registration method
    if args.method == 'standard':
        method_name = f'{args.dim}D Point CLIPPER'
        registration = ClipperPtRegistration(sigma=args.sigma, epsilon=args.epsilon, mindist=args.mindist, dim=args.dim)
    elif args.method == 'gravity':
        method_name = 'Gravity Constrained CLIPPER'
        registration = GravityConstrainedClipperReg(sigma=args.sigma, mindist=args.mindist, epsilon=args.epsilon)
    elif args.method == 'distvol':
        method_name = f'{args.dim}D Volume-based Registration'
        registration = DistVolSimReg(sigma=args.sigma, epsilon=args.epsilon, mindist=args.mindist, dim=args.dim)
    elif args.method == 'distvolgrav':
        method_name = f'Gravity Constrained Volume-based Registration'
        registration = DistVolGravityConstrained(sigma=args.sigma, epsilon=args.epsilon, mindist=args.mindist)
    else:
        assert False, "Invalid method"
    # registration = DistVolSimReg(sigma=args.sigma, epsilon=args.epsilon, vol_score_min=0.5, dist_score_min=0.5)

    # iterate over pairs of submaps and create registration results
    for i in tqdm(range(len(submaps[0]))):
        for j in (range(len(submaps[1]))):
            
            submap_intersection = circle_intersection(
                center1=submap_centers[0][i][:2], center2=submap_centers[1][j][:2], radius1=args.submap_radius, radius2=args.submap_radius
            )
            submap_area = np.pi*args.submap_radius**2
            robots_nearby_mat[i, j] = submap_intersection/submap_area

            # if show_false_positives is not set, then skip over non overlapping submaps
            if not args.show_false_positives and robots_nearby_mat[i, j] < args.submap_overlap_threshold:
                continue

            # Bring the submaps closer to the origin so that translation errors are easier to observe
            submap_i = [obj.copy() for obj in submaps[0][i]]
            submap_j = [obj.copy() for obj in submaps[1][j]]
            center_pt = np.mean([np.array(submap_centers[0][i]).reshape(-1), np.array(submap_centers[1][j]).reshape(-1)], axis=0)
            T_simplify = np.eye(4)
            T_simplify[:args.dim, 3] = -center_pt[:args.dim]

            for obj in submap_i + submap_j:
                obj.transform(T_simplify)
                
            associations = registration.register(submap_i, submap_j)
            if args.dim == 2:
                try:
                    T_align = registration.T_align(submap_i, submap_j, associations)
                    _, _, theta = transform_2_xytheta(T_align)
                    dist = np.linalg.norm(T_align[:args.dim, 3])
                except InsufficientAssociationsException:
                    theta = 180.0
                    dist = 1e6
                    associations = []

            elif args.dim == 3:
                try:
                    T_align = registration.T_align(submap_i, submap_j, associations)
                    theta = Rot.from_matrix(T_align[:3, :3]).magnitude()
                    dist = np.linalg.norm(T_align[:args.dim, 3])
                except InsufficientAssociationsException:
                    theta = 180.0
                    dist = 1e6
                    associations = []

            else:
                raise ValueError("Invalid dimension")
            
            if robots_nearby_mat[i, j] > args.submap_overlap_threshold:
                clipper_angle_mat[i, j] = np.abs(np.rad2deg(theta))
                clipper_dist_mat[i, j] = dist
            else:
                clipper_angle_mat[i, j] = np.nan
                clipper_dist_mat[i, j] = np.nan

            clipper_num_associations[i, j] = len(associations)

    # Create plots
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    fig.subplots_adjust(wspace=.3)
    fig.suptitle(method_name)

    mp = ax[0].imshow(robots_nearby_mat, vmin=0)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[0].set_title("Submaps Overlap")

    mp = ax[1].imshow(-clipper_angle_mat, vmax=0, vmin=-10)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[1].set_title("Registration Error (deg)")

    mp = ax[2].imshow(-clipper_dist_mat, vmax=0, vmin=-5.0)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[2].set_title("Registration Distance Error (m)")

    mp = ax[3].imshow(clipper_num_associations, vmin=0)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[3].set_title("Number of CLIPPER Associations")

    for i in range(4):
        ax[i].set_xlabel("submap index (robot 2)")
        ax[i].set_ylabel("submap index (robot 1)")

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
        
    if args.matrix_file:
        pkl_file = open(args.matrix_file, 'wb')
        pickle.dump([robots_nearby_mat, clipper_angle_mat, clipper_dist_mat, clipper_num_associations], pkl_file)
        pkl_file.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=2)
    parser.add_argument('--show-maps', action='store_true')
    parser.add_argument('--dim', '-d', type=int, default=3)
    parser.add_argument('--method', '-m', type=str, default='standard')
    parser.add_argument('--show-false-positives', '-s', action='store_true')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--matrix-file', type=str, default=None)
    parser.add_argument('-r', '--submap-radius', type=float, default=15.0)
    parser.add_argument('-c', '--submap-center-dist', type=float, default=15.0)
    
    args = parser.parse_args()

    # constant params
    # args.submap_radius = 15. # 20.0 for kimera multi data?
    # args.submap_center_dist = 15.0

    args.sigma = .3
    args.epsilon = .4
    args.mindist = 0.2
    # args.submaps_idx = [[32, 42], [45, 55], ]
    # args.submaps_idx = [[38, 52], [20, 35],]
    args.submaps_idx = None
    args.submap_overlap_threshold = 0.1
    
    main(args)