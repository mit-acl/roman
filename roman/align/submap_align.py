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

from roman.map.map import Submap, SubmapParams, submaps_from_roman_map, load_roman_map
from roman.align.object_registration import InsufficientAssociationsException
from roman.align.dist_reg_with_pruning import GravityConstraintError
from roman.utils import object_list_bounds, transform_rm_roll_pitch
from roman.params.submap_align_params import SubmapAlignParams, SubmapAlignInputOutput
from roman.align.results import save_submap_align_results, SubmapAlignResults

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
            # set environment variable so an individual param file is not needed
            # for each robot / run
            if sm_io.robot_env is not None:
                os.environ[sm_io.robot_env] = sm_io.robot_names[i]
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
        submap_params = SubmapParams.from_submap_align_params(sm_params)
        submap_params.use_minimal_data = True
        roman_maps = [load_roman_map(sm_io.inputs[i]) for i in range(2)]
        submaps = [submaps_from_roman_map(
            roman_maps[i], submap_params, gt_pose_data[i]) for i in range(2)]
    elif sm_io.input_type_json: # TODO: re-implement support for json files
        assert False, "Not currently supported"
        # submap_centers, submaps = load_segment_slam_submaps(sm_io.inputs, sm_params, sm_io.debug_show_maps)
        # times = [None, None]
        # trackers = [None, None]
        # submap_idxs = [None, None]

    # Registration setup
    clipper_angle_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    clipper_dist_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    clipper_num_associations = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
    robots_nearby_mat = np.zeros((len(submaps[0]), len(submaps[1])))*np.nan
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
            
            if submaps[0][i].has_gt and submaps[1][j].has_gt:
                submap_distance = norm(submaps[0][i].position_gt - submaps[1][j].position_gt)
            else:
                submap_distance = norm(submaps[0][i].position - submaps[1][j].position)
            if submap_distance < sm_params.submap_radius*2:
                robots_nearby_mat[i, j] = submap_distance

            submap_i = deepcopy(submaps[0][i])
            submap_j = deepcopy(submaps[1][j])
            if sm_params.single_robot_lc: # self loop closures
                ids_i = set([seg.id for seg in submap_i.segments])
                ids_j = set([seg.id for seg in submap_j.segments])
                common_ids = ids_i.intersection(ids_j)
                for sm in [submap_i, submap_j]:
                    to_rm = [seg for seg in sm.segments if seg.id in common_ids]
                    for seg in to_rm:
                        sm.segments.remove(seg)

            # determine correct T_ij
            if gt_pose_data[0] is not None:
                T_wi = submaps[0][i].pose_gravity_aligned_gt
            else:
                T_wi = submaps[0][i].pose_gravity_aligned
            if gt_pose_data[1] is not None:
                T_wj = submaps[1][j].pose_gravity_aligned_gt
            else:
                T_wj = submaps[1][j].pose_gravity_aligned
            T_ij = np.linalg.inv(T_wi) @ T_wj
            if not np.isnan(robots_nearby_mat[i, j]):
                relative_yaw_angle = transform_to_xyzrpy(T_ij)[5]
                submap_yaw_diff_mat[i, j] = np.abs(np.rad2deg(relative_yaw_angle))
                
            # register the submaps
            try:
                start_t = time.time()
                associations = registration.register(submap_i.segments, submap_j.segments)
                timing_list.append(time.time() - start_t)
                
                if sm_params.dim == 2:
                    T_ij_hat = registration.T_align(submap_i.segments, submap_j.segments, associations)
                    T_error = np.linalg.inv(T_ij_hat) @ T_ij
                    _, _, theta = transform_to_xytheta(T_error)
                    dist = np.linalg.norm(T_error[:sm_params.dim, 3])

                elif sm_params.dim == 3:
                    T_ij_hat = registration.T_align(submap_i.segments, submap_j.segments, associations)
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
            
            if not np.isnan(robots_nearby_mat[i, j]):
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
        submap_align_params=sm_params,
        submap_io=sm_io
    )
    save_submap_align_results(results, submaps, roman_maps)