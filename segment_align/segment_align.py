import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from robot_utils.robot_data.pose_data import PoseData
from robot_utils.transform import transform_2_xytheta

from segment_track.segment import Segment
from segment_track.tracker import Tracker

from object_map_registration.object.ellipsoid import Ellipsoid
from object_map_registration.object.object import Object
from object_map_registration.register.clipper_pt_registration import ClipperPtRegistration
from object_map_registration.register.dist_vol_sim_reg import DistVolSimReg
from object_map_registration.register.object_registration import InsufficientAssociationsException
from object_map_registration.utils import object_list_bounds

def centroid_from_points(segment: Segment):
    if segment.points is not None:
        return np.mean(segment.points, axis=0)
    else:
        return None
    
def load_pose_data(pose_file, csv=False):
    kmd_csv_options = {'cols': {'time': ['#timestamp_kf'],
                                'position': ['x', 'y', 'z'],
                                'orientation': ['qx', 'qy', 'qz', 'qw']},
                    'col_nums': {'time': [0],
                                'position': [1,2,3],
                                'orientation': [5,6,7,4]},
                    'timescale': 1e-9}
    if not csv:
        return PoseData(
            data_file=os.path.expanduser(os.path.expandvars(pose_file)),
            file_type='bag',
            topic="/acl_jackal2/kimera_vio_ros/odometry",
            time_tol=10.0,
            interp=True,
        )
    else:
        return PoseData(
            data_file=os.path.expanduser(os.path.expandvars(pose_file)),
            file_type='csv',
            time_tol=10.0,
            interp=True,
            csv_options=kmd_csv_options
        )

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

    # load pose data
    pose_data = [load_pose_data(args.pose_file1, csv=args.csv), load_pose_data(args.pose_file2, csv=args.csv)]
        
    # plot points from segments
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
                volume = segment.aabb_volume()
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

    #     fig, ax = plt.subplots()
    #     for obj in object_maps[i]:
    #         obj.plot2d(ax, color=colors[i])
        
    #     xlim, ylim = object_list_bounds(object_maps[i])
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(ylim)

    # plt.show()

    # register object maps
    # registration = ClipperPtRegistration(sigma=.3, epsilon=.6)
    # T_align = registration.T_align(object_maps[0][:100], object_maps[1][:100])
    # print(T_align)
    
    # create submaps
    ts = [np.arange(t0, tf, args.dt_grid) for t0, tf in zip(t0s, tfs)]
    clipper_angle_mat = np.empty((ts[0].shape[0] - 1, ts[1].shape[0] - 1))
    clipper_num_associations = np.empty((ts[0].shape[0] - 1, ts[1].shape[0] - 1))
    robots_nearby_mat = np.empty((ts[0].shape[0] - 1, ts[1].shape[0] - 1))
    registration = ClipperPtRegistration(sigma=args.sigma, epsilon=args.epsilon)
    # registration = DistVolSimReg(sigma=args.sigma, epsilon=args.epsilon)
    # registration = DistVolSimReg(sigma=args.sigma, epsilon=args.epsilon, vol_score_min=0.5, dist_score_min=0.5)

    # for i, ti in enumerate(tqdm(ts[0][:-1])):
    #     submap_i = [obj for obj in object_maps[0] if obj.first_seen <= ti + args.dt_map and obj.last_seen >= ti]
    
    for i, ti in enumerate(tqdm(ts[0][:-1])):
        submap_i = [obj for obj in object_maps[0] if obj.first_seen <= ti + args.dt_map and obj.last_seen >= ti]
        for j, tj in enumerate(ts[1][:-1]):
            robot_approx_dist = np.linalg.norm(pose_data[0].position(ti) - pose_data[1].position(tj))
            robots_nearby_mat[i, j] = 1 if robot_approx_dist < 30.0 else 0

            submap_j = [obj for obj in object_maps[1] if obj.first_seen <= tj + args.dt_map and obj.last_seen >= tj]
            
            if args.dim == 2:
                try:
                    T_align = registration.T_align(submap_i, submap_j)
                except InsufficientAssociationsException:
                    clipper_angle_mat[i, j] = 180.0
                    clipper_num_associations[i, j] = 0
                    continue
                _, _, theta = transform_2_xytheta(T_align)

            elif args.dim == 3:
                try:
                    T_align = registration.T_align(submap_i, submap_j)
                except InsufficientAssociationsException:
                    clipper_angle_mat[i, j] = 180.0
                    clipper_num_associations[i, j] = 0
                    continue
                theta = Rot.from_matrix(T_align[:3, :3]).magnitude()

            else:
                raise ValueError("Invalid dimension")
            
            clipper_angle_mat[i, j] = np.abs(np.rad2deg(theta))

            associations = registration.register(submap_i, submap_j)
            clipper_num_associations[i, j] = len(associations)

    fig, ax = plt.subplots()
    mp = ax.imshow(robots_nearby_mat, vmin=0)
    fig.colorbar(mp)
    ax.set_title("Robots Nearby")

    fig, ax = plt.subplots()
    mp = ax.imshow(-clipper_angle_mat, vmax=0, vmin=-20)
    fig.colorbar(mp)
    ax.set_title("Registration Angle (should be near 0 when nearby)")

    fig, ax = plt.subplots()
    mp = ax.imshow(clipper_num_associations, vmin=0)
    fig.colorbar(mp)
    ax.set_title("Number of CLIPPER Associations")

    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=2)
    
    args = parser.parse_args()

    # constant params
    args.dim = 2
    args.dt_map = 30.0
    args.dt_grid = 10.0
    # args.pose_file1 = "$WEST_POINT_2023_PATH/1117_02_map_tower_third_origin/dataset.bag"
    # args.pose_file2 = "$WEST_POINT_2023_PATH/1117_04_loc_tower_loop_far_side_loop_o3_CCW/align.bag"
    args.pose_file1 = "$KMD_GT_PATH/acl_jackal2.csv"
    args.pose_file2 = "$KMD_GT_PATH/sparkal2.csv"
    args.csv = True

    args.sigma = .3
    args.epsilon = .5
    
    main(args)