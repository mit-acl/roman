import numpy as np
import argparse
import os
import pickle
from typing import List

from robotdatapy import transform

def extract_roman_map_data(pkl_file):
    with open(os.path.expanduser(pkl_file), 'rb') as f:
        pickle_data = pickle.load(f)
        tracker, poses, times = pickle_data
        return tracker, poses, times

def create_information_matrix(t_std, r_std):
    I_t = 1 / (t_std**2)
    I_r = 1 / (r_std**2)
    I = np.diag([I_t, I_t, I_t, I_r, I_r, I_r])
    return I

def extract_odom_g2o(poses: List[np.array], times: List[float], I: np.array, 
                     min_keyframe_dist: bool = None):
    """
    Turns odometry data from saved ROMAN pickle file into g2o/time format.

    Args:
        poses (List[np.array]): List of poses.
        times (List[float]): List of times.
        I (np.array): Information matrix.
        min_keyframe_dist (bool, optional): Minimum distance between keyframes. No min if None. 
            Defaults to None.
    """
    edge_lines = []
    vertex_lines = []
    selected_times = []

    next_i = 0
    idx_list = []
    for i in range(len(poses) - 1):
        if min_keyframe_dist is not None and i < next_i:
            continue
        idx_list.append(i)
        T_w1 = poses[i]
        T_w2 = poses[i + 1]
        if min_keyframe_dist is not None:
            j = i
            while j < len(poses) - 1:
                j += 1
                T_w2 = poses[j]
                # continue until the next keyframe is far enough
                if np.linalg.norm(T_w1[:3, 3] - T_w2[:3, 3]) > min_keyframe_dist:
                    break
            next_i = j
            
        T_12 = np.linalg.inv(T_w1) @ T_w2
        t, q = transform.transform_to_xyz_quat(T_12, separate=True)
        new_line = f"EDGE_SE3:QUAT {len(idx_list) - 1} {len(idx_list)} \t\t"
        new_line += f"{t[0]} {t[1]} {t[2]} \t\t"
        new_line += f"{q[0]} {q[1]} {q[2]} {q[3]} \t\t"
        for ii in range(6):
            for jj in range(6):
                if jj < ii:
                    continue
                new_line += f"{I[ii, jj]} "
            new_line += "\t\t"
        new_line += "\n"
        edge_lines.append(new_line)
        
        # Make sure that the last pose is included
        if min_keyframe_dist is not None:
            if next_i == len(poses) - 1:
                idx_list.append(j)
        elif i == len(poses) - 2:
            idx_list.append(i + 1)
        
    for new_i, i in enumerate(idx_list):
        pose = poses[i]
        t, q = transform.transform_to_xyz_quat(pose, separate=True)
        vertex_lines += f"VERTEX_SE3:QUAT {new_i} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n"
        selected_times.append(times[i])
        
    return vertex_lines, edge_lines, selected_times

def roman_map_pkl_to_g2o(
    pkl_file: str,
    g2o_file: str,
    time_file: str = None,
    robot_id: int = 0,
    min_keyframe_dist: float = None,
    t_std: float = 0.005,
    r_std: float = np.deg2rad(0.025),
    verbose: bool = False
):

    # setup information matrix
    I = create_information_matrix(t_std, r_std)
    
    # open input ROMAN map pkl file
    _, poses, times = extract_roman_map_data(pkl_file)
    
    # extract g2o data
    vertex_lines, edge_lines, selected_times = \
        extract_odom_g2o(poses, times, I, min_keyframe_dist)
            
    with open(os.path.expanduser(g2o_file), 'w') as f:
        for line in vertex_lines + edge_lines:
            f.write(line)    
        f.close()

    if verbose:
        print(f"Saved g2o to {os.path.abspath(g2o_file)}")
    
    if time_file is None:
        return
    
    assert times is not None, "No time data found in pickle file."
    
    with open(os.path.expanduser(time_file), 'w') as f:
        for i, time in enumerate(selected_times):
            f.write(f"{robot_id} {i} {int(time*1e9)} xxx\n")
        f.close()
        
    if verbose:
        print(f"Saved time data to {os.path.abspath(time_file)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SegTrack to g2o format.')
    parser.add_argument('input', type=str, help='Input SegTrack file.')
    parser.add_argument('output', type=str, help='Output g2o file.')
    parser.add_argument('-t', '--output-time', action='store_true', help='Output timing information.')
    parser.add_argument('-f', '--time-file', type=str, default=None, 
                        help='Time file. Saves with same name as g2o file with time.txt extension if this is not set.')
    parser.add_argument('-n', '--robot-id', type=int, default=0, help='Robot ID.')
    parser.add_argument('-d', '--min-keyframe-dist', type=float, default=None, help="Minimum distance between keyframes.")
    parser.add_argument('-s', '--std', default=[.01, .02], nargs=2, type=float, help='Standard deviation of translation and rotation.')
    args = parser.parse_args()

    args.t_std = args.std[0]
    args.r_std = np.deg2rad(args.std[1])
    
    if args.time_file is None and args.output_time:
        args.time_file = args.output.replace('.g2o', '_time.txt')
    
    roman_map_pkl_to_g2o(
        args.input,
        args.output,
        args.time_file,
        args.robot_id,
        args.min_keyframe_dist,
        args.t_std,
        args.r_std,
        verbose=True
    )