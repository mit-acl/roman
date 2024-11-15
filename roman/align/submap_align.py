import argparse
import numpy as np
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation as Rot
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List
import open3d as o3d
import clipperpy
import time
import json
from copy import deepcopy
import yaml

from robotdatapy.data.pose_data import PoseData
from robotdatapy.transform import transform_to_xytheta, transform_to_xyz_quat, \
    transform_to_xyzrpy
from robotdatapy.geometry import circle_intersection

from roman.object.segment import Segment
from roman.map.tracker import Tracker

from roman.object.object import Object
from roman.object.segment import SegmentMinimalData
from roman.align.object_registration import InsufficientAssociationsException
from roman.align.dist_reg_with_pruning import GravityConstraintError
from roman.utils import object_list_bounds, transform_rm_roll_pitch
from roman.align.params import SubmapAlignParams, SubmapAlignInputOutput
from roman.align.results import save_submap_align_results, SubmapAlignResults

OVERLAP_EPS = 0.1

def submap_centers_from_poses(poses: List[np.array], dist: float, times: List[float]=None, time_threshold: float=np.inf):
    """
    From a series of poses, generate a list of positions on the trajectory that are dist apart.

    Args:
        poses (List[np.array, shape=(3,3 or 4,4)]): list of poses
        dist (float): Distance between submap centers
    
    Returns:
        np.array, shape=(3,n)]: List of submap centers
    """
    assert times is None or len(poses) == len(times)
    centers = []
    center_times = []
    center_idxs = []

    for i, (pose, t) in enumerate(zip(poses, times)):
        if i == 0 or np.linalg.norm(pose[:-1,-1] - centers[-1][:-1,-1]) > dist \
            or (times is not None and t - center_times[-1] > time_threshold):
            centers.append(pose)
            center_idxs.append(i)
            if times is not None:
                center_times.append(t)
    return centers, center_times, center_idxs

def submaps_from_maps(object_map: List[Object], centers: List[np.array],
                      center_times: List[float]=None, submap_align_params: SubmapAlignParams = SubmapAlignParams()):
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
    for i, center in enumerate(centers):
        if center_times is not None:
            t = center_times[i]
            tm1 = center_times[i-1] if i > 0 else center_times[i]
            tp1 = center_times[i+1] if i < len(centers) - 1 else center_times[i]
            meets_time_thresh = \
                lambda obj: not (obj.first_seen > tp1 + submap_align_params.submap_center_time
                    or obj.last_seen < tm1 - submap_align_params.submap_center_time)
        else:
            meets_time_thresh = lambda obj: True
        # submaps is originally a collection of objects that are within a radius from the center
        submap = [deepcopy(obj) for obj in object_map if \
                  np.linalg.norm(obj.center.flatten()[:2] - center[:2,3]) < submap_align_params.submap_radius \
                  and meets_time_thresh(obj)]
        
        T_odom_center = transform_rm_roll_pitch(center)
        T_center_odom = np.linalg.inv(T_odom_center)
        for obj in submap:
            obj.transform(T_center_odom)

        if submap_align_params.submap_max_size is not None:
            submap_sorted_by_dist = sorted(submap, key=lambda obj: np.linalg.norm(obj.center.flatten()[:2]))
            submap = submap_sorted_by_dist[:submap_align_params.submap_max_size]

        submaps.append(submap)
    return submaps

def create_submaps(
    pickle_files: List[str], 
    submap_align_params: SubmapAlignParams=SubmapAlignParams(), 
    show_maps: bool=False
):    
    colors = ['maroon', 'blue']
    poses = []
    times = []
    trackers = []
    object_maps = [[], []]
    
    # extract pickled data
    for i, pickle_file in enumerate(pickle_files):
        with open(os.path.expanduser(pickle_file), 'rb') as f:
            pickle_data = pickle.load(f)
            
            tr, p, t = pickle_data
            times.append(t)
            poses.append(p)
            trackers.append(tr)

    # create objects from segments
    for i, tracker in enumerate(trackers):
        segment: Segment
        for segment in tracker.segments + tracker.inactive_segments + tracker.segment_graveyard:
            if submap_align_params.use_object_bottom_middle:
                segment.set_center_ref("bottom_middle")
            if segment.points is not None and len(segment.points) > 0:
                try:
                    object_maps[i].append(segment.minimal_data())
                except Exception as e:
                    print(e)
                    continue

    # create submaps
    submap_centers, submap_times, submap_idxs = \
        zip(*[submap_centers_from_poses(pose_list, submap_align_params.submap_center_dist, ts) \
            for pose_list, ts in zip(poses, times)])
    submaps = [submaps_from_maps(object_map, center_list, time_list, submap_align_params) \
        for object_map, center_list, time_list \
        in zip(object_maps, submap_centers, submap_times)]
    
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
                ax.add_patch(plt.Circle(center[:2,3], submap_align_params.submap_radius, color='black', fill=False))
            ax.set_aspect('equal')
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        plt.show()
        exit(0)
        
    return submap_centers, submaps, poses, times, trackers, submap_idxs

def load_segment_slam_submaps(json_files: List[str], 
        sm_params: SubmapAlignParams=SubmapAlignParams(), show_maps=False):
    submaps = []
    submap_centers = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            smcs = []
            sms = []
            objs = {}
            
            data = json.load(f)
            for seg in data['segments']:
                centroid = np.array([seg['centroid_odom']['x'], seg['centroid_odom']['y'], seg['centroid_odom']['z']])[:sm_params.dim]
                new_obj = SegmentMinimalData(
                    id=seg['segment_index'],
                    center=centroid,
                    volume=seg['shape_attributes']['volume'],
                    linearity=seg['shape_attributes']['linearity'],
                    planarity=seg['shape_attributes']['planarity'],
                    scattering=seg['shape_attributes']['scattering'],
                    extent=None,
                    semantic_descriptor=None
                )
                objs[seg['segment_index']] = new_obj
                
            for submap in data['submaps']:
                center = np.eye(4)
                center[:3,3] = np.array([submap['T_odom_submap']['tx'], submap['T_odom_submap']['ty'], submap['T_odom_submap']['tz']])
                center[:3,:3] = Rot.from_quat([submap['T_odom_submap']['qx'], submap['T_odom_submap']['qy'], submap['T_odom_submap']['qz'], submap['T_odom_submap']['qw']]).as_matrix()
                sm = [deepcopy(objs[idx]) for idx in submap['segment_indices']]

                # Transform objects to be centered at the submap center
                T_submap_world = np.eye(4) # transformation to move submap from world frame to centered submap frame
                T_submap_world[:sm_params.dim, 3] = -center[:sm_params.dim, 3]
                for obj in sm:
                    obj.transform(T_submap_world)

                smcs.append(center)
                sms.append(sm)
                
            submap_centers.append(smcs)
            submaps.append(sms)
    if show_maps:
        for i in range(2):
            for submap in submaps[i]:
                fig, ax = plt.subplots()
                for obj in submap:
                    obj.plot2d(ax, color='blue')
                
                bounds = object_list_bounds(submap)
                if len(bounds) == 3:
                    xlim, ylim, _ = bounds
                else:
                    xlim, ylim = bounds

                # ax.plot([position[0] for position in submap_centers[i]], [position[1] for position in submap_centers[i]], 'o', color='black')
                ax.set_aspect('equal')
                
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            plt.show()
        exit(0)
    return submap_centers, submaps
        

def submap_align(sm_params: SubmapAlignParams, sm_io: SubmapAlignInputOutput):
    """
    Breaks maps into submaps and attempts to align each submap from one map with each submap from the second map.

    Args:
        sm_params (SubmapAlignParams): Aignment (loop closure) params.
        sm_io (SubmapAlignInputOutput): Input/output specifications.
    """
    assert sm_io.input_type_json or sm_io.input_type_pkl, "Invalid input type"
    assert sm_io.input_type_json != sm_io.input_type_pkl, "Only one input type allowed"
    
    gt_pose_data = [None, None]
    
    # load ground truth pose data
    for i, yaml_file in enumerate(sm_io.input_gt_pose_yaml):
        if yaml_file is not None:
            # load yaml file
            with open(os.path.expanduser(yaml_file), 'r') as f:
                gt_pose_args = yaml.safe_load(f)
            if gt_pose_args['type'] == 'bag':
                gt_pose_data[i] = PoseData.from_bag(**{k: v for k, v in gt_pose_args.items() if k != 'type'})
            elif gt_pose_args['type'] == 'csv':
                gt_pose_data[i] = PoseData.from_csv(**{k: v for k, v in gt_pose_args.items() if k != 'type'})
            elif gt_pose_args['type'] == 'bag_tf':
                gt_pose_data[i] = PoseData.from_bag_tf(**{k: v for k, v in gt_pose_args.items() if k != 'type'})
            else:
                raise ValueError("Invalid pose data type")
    
    if sm_io.input_type_pkl:
        submap_centers, submaps, poses, times, trackers, submap_idxs = \
            create_submaps(sm_io.inputs, sm_params, sm_io.debug_show_maps)
    elif sm_io.input_type_json: # TODO: re-implement support for json files
        submap_centers, submaps = load_segment_slam_submaps(sm_io.inputs, sm_params, sm_io.debug_show_maps)
        times = [None, None]
        trackers = [None, None]
        submap_idxs = [None, None]

    # Registration setup
    clipper_angle_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    clipper_dist_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    clipper_num_associations = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    robots_nearby_mat = np.empty((len(submaps[0]), len(submaps[1])))
    clipper_percent_associations = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    submap_yaw_diff_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    timing_list = []
    
    T_ij_mat = np.zeros((len(submaps[0]), len(submaps[1]), 4, 4))*np.nan
    T_ij_hat_mat = np.zeros((len(submaps[0]), len(submaps[1]), 4, 4))*np.nan
    associated_objs_mat = [[[] for _ in range(len(submaps[1]))] for _ in range(len(submaps[0]))] # cannot be numpy array since each element is a different sized array

    # Registration method
    registration = sm_params.get_object_registration()


    # iterate over pairs of submaps and create registration results
    for i in tqdm(range(len(submaps[0]))):
        for j in (range(len(submaps[1]))):
            
            if gt_pose_data != [None, None]:
                sc_gt = [gt_pose_data[0].pose(times[0][submap_idxs[0][i]]), gt_pose_data[1].pose(times[1][submap_idxs[1][j]])]
            else:
                sc_gt = [submap_centers[0][i], submap_centers[1][j]]
            submap_intersection = circle_intersection(
                center1=sc_gt[0][:2,3], center2=sc_gt[1][:2,3], radius1=sm_params.submap_radius, radius2=sm_params.submap_radius
            )
            submap_area = np.pi*sm_params.submap_radius**2
            robots_nearby_mat[i, j] = submap_intersection/submap_area

            submap_i = deepcopy(submaps[0][i])
            submap_j = deepcopy(submaps[1][j])
            if sm_params.single_robot_lc: # self loop closures
                ids_i = set([obj.id for obj in submap_i])
                ids_j = set([obj.id for obj in submap_j])
                common_ids = ids_i.intersection(ids_j)
                for sm in [submap_i, submap_j]:
                    to_rm = [obj for obj in sm if obj.id in common_ids]
                    for obj in to_rm:
                        sm.remove(obj)

            # determine correct T_ij
            if gt_pose_data[0] is not None:
                T_wi = transform_rm_roll_pitch(gt_pose_data[0].pose(times[0][submap_idxs[0][i]]))
            else:
                T_wi = transform_rm_roll_pitch(submap_centers[0][i])
            if gt_pose_data[1] is not None:
                T_wj = transform_rm_roll_pitch(gt_pose_data[1].pose(times[1][submap_idxs[1][j]]))
            else:
                T_wj = transform_rm_roll_pitch(submap_centers[1][j])
            T_ij = np.linalg.inv(T_wi) @ T_wj
            if robots_nearby_mat[i, j] > OVERLAP_EPS:
                relative_yaw_angle = transform_to_xyzrpy(T_ij)[5]
                submap_yaw_diff_mat[i, j] = np.abs(np.rad2deg(relative_yaw_angle))
                
            # register the submaps=
            try:
                start_t = time.time()
                associations = registration.register(submap_i, submap_j)
                timing_list.append(time.time() - start_t)
                
                if sm_params.dim == 2:
                    T_ij_hat = registration.T_align(submap_i, submap_j, associations)
                    T_error = np.linalg.inv(T_ij_hat) @ T_ij
                    _, _, theta = transform_to_xytheta(T_error)
                    dist = np.linalg.norm(T_error[:sm_params.dim, 3])

                elif sm_params.dim == 3:
                    T_ij_hat = registration.T_align(submap_i, submap_j, associations)
                    if sm_params.force_rm_upside_down:
                        xyzrpy = transform_to_xyzrpy(T_ij_hat)
                        if np.abs(xyzrpy[3]) > np.deg2rad(90.) or np.abs(xyzrpy[4]) > np.deg2rad(90.):
                            raise GravityConstraintError
                    if sm_params.force_rm_lc_roll_pitch:
                        T_ij_hat = transform_rm_roll_pitch(T_ij_hat)
                    T_error = np.linalg.inv(T_ij_hat) @ T_ij
                    theta = Rot.from_matrix(T_error[:3, :3]).magnitude()
                    dist = np.linalg.norm(T_error[:sm_params.dim, 3])
                else:
                    raise ValueError("Invalid dimension")
                
            except (InsufficientAssociationsException, GravityConstraintError) as ex:
                timing_list.append(time.time() - start_t)
                T_ij_hat = np.zeros((4, 4))*np.nan
                theta = 180.0
                dist = 1e6
                associations = []
            
            if robots_nearby_mat[i, j] > OVERLAP_EPS:
                clipper_angle_mat[i, j] = np.abs(np.rad2deg(theta))
                clipper_dist_mat[i, j] = dist
            else:
                clipper_angle_mat[i, j] = np.nan
                clipper_dist_mat[i, j] = np.nan

            clipper_num_associations[i, j] = len(associations)
            clipper_percent_associations[i, j] = len(associations) / np.mean([len(submap_i), len(submap_j)])
            
            T_ij_mat[i, j] = T_ij
            T_ij_hat_mat[i, j] = T_ij_hat
            associated_objs_mat[i][j] = associations

    # save results
    results = SubmapAlignResults(
        robots_nearby_mat=robots_nearby_mat,
        clipper_angle_mat=clipper_angle_mat,
        clipper_dist_mat=clipper_dist_mat,
        clipper_num_associations=clipper_num_associations,
        submap_yaw_diff_mat=submap_yaw_diff_mat,
        T_ij_mat=T_ij_mat,
        T_ij_hat_mat=T_ij_hat_mat,
        associated_objs_mat=associated_objs_mat,
        timing_list=timing_list,
    )
    save_submap_align_results(sm_params, sm_io, submaps, results, trackers, times, submap_idxs, submap_centers)