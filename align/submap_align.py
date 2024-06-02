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
from object_map_registration.register.dist_feature_sim_reg import DistOnlyReg, DistVolReg
from object_map_registration.register.dist_reg_with_pruning import DistRegWithPruning, GravityConstraintError
from object_map_registration.register.object_registration import InsufficientAssociationsException
from object_map_registration.utils import object_list_bounds
    
# TODO: right now, we don't account for a when a single robot revisits the same place - 
# in this case, there will be objects mapped from both the first and second visit in 
# a single submap. Submaps should then have some temporal distinction as well, but that 
# would require that the pose list also have timing, which is not the case right now.    

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
        # submaps is originally a collection of objects that are within a radius from the center
        submap = [obj.copy() for obj in object_map if np.linalg.norm(obj.center.flatten()[:2] - center[:2]) < radius]
        
        # Transform objects to be centered at the submap center
        T_submap_world = np.eye(4) # transformation to move submap from world frame to centered submap frame
        T_submap_world[:args.dim, 3] = -center[:args.dim]
        for obj in submap:
            obj.transform(T_submap_world)

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
                # find volume once for each object so does not have to be recomputed each time
                new_obj.volume
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
    if args.ambiguity:
        ambiguity_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    if args.output_viz_file:
        T_ij_mat = np.zeros((len(submaps[0]), len(submaps[1]), 4, 4))*np.nan
        T_ij_hat_mat = np.zeros((len(submaps[0]), len(submaps[1]), 4, 4))*np.nan
        associated_objs_mat = [[[] for _ in range(len(submaps[1]))] for _ in range(len(submaps[0]))] # cannot be numpy array since each element is a different sized array

    # Registration method
    if args.method == 'standard':
        method_name = f'{args.dim}D Point CLIPPER'
        registration = DistOnlyReg(sigma=args.sigma, epsilon=args.epsilon, mindist=args.mindist, pt_dim=args.dim)
    elif args.method == 'gravity':
        method_name = 'Gravity Constrained CLIPPER'
        registration = DistOnlyReg(sigma=args.sigma, mindist=args.mindist, epsilon=args.epsilon, use_gravity=True)
    elif args.method == 'distvol':
        method_name = f'{args.dim}D Volume-based Registration'
        registration = DistVolReg(sigma=args.sigma, epsilon=args.epsilon, mindist=args.mindist, pt_dim=args.dim, volume_epsilon=args.epsilon_volume)
    elif args.method == 'distvolgrav':
        method_name = f'Gravity Constrained Volume-based Registration'
        registration = DistVolReg(sigma=args.sigma, epsilon=args.epsilon, mindist=args.mindist, use_gravity=True, volume_epsilon=args.epsilon_volume)
    elif args.method == 'prunevol':
        method_name = f'{args.dim}D Volume-based Pruning'
        registration = DistRegWithPruning(sigma=args.sigma, epsilon=args.epsilon, mindist=args.mindist, dim=args.dim, volume_epsilon=args.epsilon_volume, use_gravity=False)
    elif args.method == 'prunevolgrav':
        method_name = f'Gravity Filtered Volume-based Pruning'
        registration = DistRegWithPruning(sigma=args.sigma, epsilon=args.epsilon, mindist=args.mindist, dim=args.dim, volume_epsilon=args.epsilon_volume, use_gravity=True)
    elif args.method == 'prunegrav':
        method_name = f'Gravity Filtered Pruning'
        registration = DistRegWithPruning(sigma=args.sigma, epsilon=args.epsilon, mindist=args.mindist, dim=args.dim, use_gravity=True)
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

            submap_i = submaps[0][i]
            submap_j = submaps[1][j]

            # determine correct T_ij
            T_wi = np.eye(4); T_wi[:args.dim, 3] = submap_centers[0][i]
            T_wj = np.eye(4); T_wj[:args.dim, 3] = submap_centers[1][j]
            T_ij = np.linalg.inv(T_wi) @ T_wj
                
            # register the submaps (run multiple times if ambiguity is set)
            if args.ambiguity:
                solutions = registration.mno_clipper(submap_i, submap_j, num_solutions=2)
                ambiguity_mat[i, j] = np.min([solutions[k][1] for k in range(2)]) / np.max([solutions[k][1] for k in range(2)])
                associations = solutions[0][0]

            try:
                associations = registration.register(submap_i, submap_j)
                
                if args.dim == 2:
                    T_ij_hat = registration.T_align(submap_i, submap_j, associations)
                    T_error = np.linalg.inv(T_ij_hat) @ T_ij
                    _, _, theta = transform_2_xytheta(T_error)
                    dist = np.linalg.norm(T_error[:args.dim, 3])

                elif args.dim == 3:
                    T_ij_hat = registration.T_align(submap_i, submap_j, associations)
                    T_error = np.linalg.inv(T_ij_hat) @ T_ij
                    theta = Rot.from_matrix(T_error[:3, :3]).magnitude()
                    dist = np.linalg.norm(T_error[:args.dim, 3])
                else:
                    raise ValueError("Invalid dimension")
                
            except (InsufficientAssociationsException, GravityConstraintError) as ex:
                T_ij_hat = np.zeros((4, 4))*np.nan
                theta = 180.0
                dist = 1e6
                associations = []
            
            if robots_nearby_mat[i, j] > args.submap_overlap_threshold:
                clipper_angle_mat[i, j] = np.abs(np.rad2deg(theta))
                clipper_dist_mat[i, j] = dist
            else:
                clipper_angle_mat[i, j] = np.nan
                clipper_dist_mat[i, j] = np.nan

            clipper_num_associations[i, j] = len(associations)
            
            if args.output_viz_file:
                T_ij_mat[i, j] = T_ij
                T_ij_hat_mat[i, j] = T_ij_hat
                associated_objs_mat[i][j] = associations

    # Create plots
    if args.ambiguity:
        fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    else:
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

    if args.ambiguity:
        mp = ax[4].imshow(ambiguity_mat, vmin=0, vmax=1.0)
        fig.colorbar(mp, fraction=0.03, pad=0.04)
        ax[4].set_title("Ambiguity")

    for i in range(len(ax)):
        ax[i].set_xlabel("submap index (robot 2)")
        ax[i].set_ylabel("submap index (robot 1)")

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
        
    # for saving matrix results instead of image
    if args.matrix_file:
        pkl_file = open(args.matrix_file, 'wb')
        if not args.ambiguity:
            pickle.dump([robots_nearby_mat, clipper_angle_mat, clipper_dist_mat, clipper_num_associations], pkl_file)
        else:
            pickle.dump([robots_nearby_mat, clipper_angle_mat, clipper_dist_mat, clipper_num_associations, ambiguity_mat], pkl_file)
        pkl_file.close()
        
    # stores the submaps, associated objects, ground truth object overlap, and ground truth and estimated submap transformations
    if args.output_viz_file:
        submaps_pkl = [[[obj.to_pickle() for obj in sm] for sm in sms] for sms in submaps]
        pkl_file = open(args.output_viz_file, 'wb')
        # submaps for robot i, submaps for robot j, associated object indices, matrix of how much overlap is between submaps, ground truth T_ij, estimated T_ij
        pickle.dump([submaps_pkl[0], submaps_pkl[1], associated_objs_mat, robots_nearby_mat, T_ij_mat, T_ij_hat_mat], pkl_file)
        pkl_file.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=2)
    parser.add_argument('--show-maps', action='store_true', help="Shows the submaps plotted as 2D points to help with submap sizing")
    parser.add_argument('--dim', '-d', type=int, default=3, help="2 or 3 - desired dimension to use for object alignment")
    parser.add_argument('--method', '-m', type=str, default='standard', help="Method to use for registration: standard, gravity, distvol, distvolgrav, prunevol, prunevolgrav, prunegrav")
    parser.add_argument('--show-false-positives', '-s', action='store_true', help="Run alignment for submaps that do not overlap")
    parser.add_argument('--output', '-o', type=str, default=None, help="Path to save output plot")
    parser.add_argument('--matrix-file', type=str, default=None, help="Path to save matrix pickle file")
    parser.add_argument('--output-viz-file', type=str, default=None, help="Path to save output visualization file")
    parser.add_argument('-r', '--submap-radius', type=float, default=15.0, help="Radius of submaps")
    parser.add_argument('-c', '--submap-center-dist', type=float, default=15.0, help="Distance between submap centers")
    parser.add_argument('--ambiguity', '-a', action='store_true', help="Create ambiguity matrix plot")
    parser.add_argument('--submaps-idx', type=int, nargs=4, default=None, help="Specify submap indices to use")
    
    # registration params
    parser.add_argument('--sigma', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--mindist', type=float, default=0.2)
    parser.add_argument('--epsilon-volume', type=float, default=0.2)

    args = parser.parse_args()

    # constant params
    # args.submap_radius = 15. # 20.0 for kimera multi data?
    # args.submap_center_dist = 15.0

    if args.submaps_idx is not None:
        args.submaps_idx = [args.submaps_idx[:2], args.submaps_idx[2:]]
    # args.submaps_idx = None
    # args.submaps_idx = [[32, 42], [45, 55], ]
    # args.submaps_idx = [[38, 52], [20, 35],] # sparkal2/1 opposite direction
    # args.submaps_idx = [[0, 20], [0, 20],] # sparkal2/1 same direction
    # args.submaps_idx = [[4, 18], [13, 28]] # acl_jackal2/sparkal1 same direction
    # args.submaps_idx = [[4, 11], [60, 67]] # acl_jackal2/sparkal1 90 degrees crossed
    args.submap_overlap_threshold = 0.1
    
    main(args)