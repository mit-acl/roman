import numpy as np
import gtsam
from typing import List, Tuple, Dict
import os
import yaml

from robotdatapy.data.pose_data import PoseData

def time_vertex_mapping(time_file: int, robot_id: int = None, use_gtsam_idx: bool = False) -> Dict[int, float]:
    with open(time_file, 'r') as f:
        time_lines = f.readlines()
    time_lines = [line.strip().split() for line in time_lines]
    # map each index to a time for the desired robot
    times = {
                int(line[1]) if not use_gtsam_idx else 
                gtsam.symbol(chr(int(line[0]) + ord('a')), int(line[1])): 
                float(line[2])*1e-9 for line in time_lines
                if (int(line[0]) == robot_id or robot_id is None)
            }
    return times


def g2o_and_time_to_pose_data(g2o_file: str, time_file: str, robot_id: int = None) -> PoseData:
    with open(g2o_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    robot_gtsam_char = chr(ord('a') + robot_id) if robot_id is not None else None
    vertices = [line for line in lines if line[0] == 'VERTEX_SE3:QUAT']
    positions = dict()
    quats = dict()
    
    for line in vertices:

        if robot_gtsam_char is not None: # accept all odometry if robot_id is None
            if gtsam.Symbol(int(line[1])).chr() != ord(robot_gtsam_char): # accept only odometry for robot_id
                continue
            vertex_idx = gtsam.Symbol(int(line[1])).index()
        else:
            vertex_idx = int(line[1])
        
        positions[vertex_idx] = np.array([float(x) for x in line[2:5]])
        quats[vertex_idx] = np.array([float(x) for x in line[5:9]])

    assert len(positions) > 0, "No vertices found in g2o file"

    times = time_vertex_mapping(time_file, robot_id)
    
    indices = sorted(list(times.keys()))
    assert indices == sorted(list(positions.keys())), \
        f"Indices in time file and g2o file do not match: \n" + \
        f"g2o file: {g2o_file}, time file: {time_file} \n" + \
        f"{indices[:10]} {indices[-10:]} \n" + \
        f"{sorted(list(positions.keys()))[:10]} {sorted(list(positions.keys()))[-10:]}"
    
    return PoseData(
        times=[times[i] for i in indices],
        positions=[positions[i] for i in indices],
        orientations=[quats[i] for i in indices],
        interp=False
    )

def concatentate_pose_data(pose_data: List[PoseData]) -> PoseData:
    for i, pd in enumerate(pose_data):
        if i == 0:
            times = pd.times
            positions = pd.positions
            orientations = pd.orientations
        else:
            times = np.concatenate((times, pd.times - pd.t0 + times[-1] + 1.0))
            positions = np.concatenate((positions, pd.positions))
            orientations = np.concatenate((orientations, pd.orientations))
    return PoseData(times=times, positions=positions, orientations=orientations, interp=False)

def combine_multi_est_and_gt_pose_data(est: List[PoseData], gt: List[PoseData]) -> Tuple[PoseData, PoseData]:
    assert len(est) == len(gt), "Number of estimated and ground truth datasets do not match"

    # first, make sure the start and end times are exactly the same
    for est_i, gt_i in zip(est, gt):
        if est_i.t0 < gt_i.t0:
            gt_i.times = np.concatenate(([est_i.t0], gt_i.times))
            gt_i.positions = np.concatenate(([gt_i.positions[0]], gt_i.positions))
            gt_i.orientations = np.concatenate(([gt_i.orientations[0]], gt_i.orientations))
        elif est_i.t0 > gt_i.t0:
            est_i.times = np.concatenate(([gt_i.t0], est_i.times))
            est_i.positions = np.concatenate(([est_i.positions[0]], est_i.positions))
            est_i.orientations = np.concatenate(([est_i.orientations[0]], est_i.orientations))
        
        if est_i.tf < gt_i.tf:
            est_i.times = np.concatenate((est_i.times, [gt_i.tf]))
            est_i.positions = np.concatenate((est_i.positions, [est_i.positions[-1]]))
            est_i.orientations = np.concatenate((est_i.orientations, [est_i.orientations[-1]]))
        elif est_i.tf > gt_i.tf:
            gt_i.times = np.concatenate((gt_i.times, [est_i.tf]))
            gt_i.positions = np.concatenate((gt_i.positions, [gt_i.positions[-1]]))
            gt_i.orientations = np.concatenate((gt_i.orientations, [gt_i.orientations[-1]]))
    
    return concatentate_pose_data(est), concatentate_pose_data(gt)

def load_gt_pose_data(gt_file):
    if 'csv' in gt_file:
        return PoseData.from_kmd_gt_csv(gt_file) 
    with open(os.path.expanduser(gt_file), 'r') as f:
        gt_pose_args = yaml.safe_load(f)
    if gt_pose_args['type'] == 'bag':
        return PoseData.from_bag(**{k: v for k, v in gt_pose_args.items() if k != 'type'})
    elif gt_pose_args['type'] == 'csv':
        return PoseData.from_csv(**{k: v for k, v in gt_pose_args.items() if k != 'type'})
    elif gt_pose_args['type'] == 'bag_tf':
        return PoseData.from_bag_tf(**{k: v for k, v in gt_pose_args.items() if k != 'type'})
    else:
        raise ValueError("Invalid pose data type")

def gt_csv_est_g2o_to_pose_data(est_g2o_file: str, est_time_file: str, gt_csv_files: Dict[int, str]) -> Tuple[PoseData, PoseData]:
    """
    Generates two comparable PoseData objects from ground truth and estimated multi-robot poses.
    Designed for Kimera-Multi dataset where ground truth is stored in a csv file and estimated poses
    are stored as g2o files.

    Args:
        est_g2o_file (str): File path to the estimated poses in g2o format.
        est_time_file (str): File path to the estimated time file.
        gt_csv_files (Dict[int, str]): Mapping from robot_id (with 0 corresponding to 'a' in gtsam g2o file)
            to the corresponding ground truth csv file.

    Returns:
        Tuple[PoseData, PoseData]: Estimated and ground truth PoseData objects
    """
    
    pose_data_gt = [load_gt_pose_data(gt_csv_files[i])
                    for i in sorted(gt_csv_files.keys())]
    pose_data_est = [g2o_and_time_to_pose_data(est_g2o_file, est_time_file, i)
                     for i in sorted(gt_csv_files.keys())]
    
    return combine_multi_est_and_gt_pose_data(pose_data_est, pose_data_gt)
