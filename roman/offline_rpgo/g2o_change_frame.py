import numpy as np
from typing import List
import os

from robotdatapy import transform

# def g2o_change_frame(g2o: str | List[str], T_postmultiply: np.array) -> List[str]:
def g2o_change_frame(g2o: any, T_postmultiply: np.array) -> List[str]:
    """
    Change the frame of trajectory in g2o format by multiplying the poses by T_postmultiply.

    Args:
        g2o (str | List[str]): Path to the g2o file or list of g2o lines
        T_postmultiply (np.array): Transformation matrix to postmultiply to the poses of the input
    
    Returns:
        List[str]: List of edited g2o lines
    """
    if isinstance(g2o, str):
        with open(os.path.expanduser(os.path.expandvars(g2o)), 'r') as f:
            g2o_lines = f.readlines()
    else:
        g2o_lines = g2o
    
    new_g2o_lines = []
    for line in g2o_lines:
        if line.startswith('VERTEX_SE3:QUAT'):
            _, idx, *pose_xyzquat = line.split()
            pose_xyzquat = np.array([float(p) for p in pose_xyzquat])
            pose_transform = transform.xyz_quat_to_transform(pose_xyzquat[:3], pose_xyzquat[3:])
            new_pose_xyzquat = transform.transform_to_xyz_quat(pose_transform @ T_postmultiply)
            new_g2o_lines.append(f'VERTEX_SE3:QUAT {idx} {" ".join([str(el) for el in new_pose_xyzquat.tolist()])}\n')
        elif line.startswith('EDGE_SE3:QUAT'):
            _, idx1, idx2, *pose_and_info = line.split()
            pose_xyzquat = np.array([float(p) for p in pose_and_info[:7]])
            info = pose_and_info[7:]
            pose_transform = transform.xyz_quat_to_transform(pose_xyzquat[:3], pose_xyzquat[3:])

            # pose_transform used to be T_1_2
            # now, T_world_1 = T_world_1 @ T_1_1newframe
            #                              where T_1_1newframe = T_postmultiply
            # We need T_1newframe_2newframe
            # T_1newframe_2newframe = T_1newframe_1 @ T_1_2 @ T_2_2newframe
            # T_1newframe_2newframe = inv(T_postmultiply) @ T_1_2 @ T_postmultiply
            new_pose_xyzquat = transform.transform_to_xyz_quat(np.linalg.inv(T_postmultiply) @ pose_transform @ T_postmultiply)
            new_g2o_lines.append(f'EDGE_SE3:QUAT {idx1} {idx2} {" ".join([str(el) for el in new_pose_xyzquat.tolist()])} {" ".join(info)}\n')
        else:
            new_g2o_lines.append(line)

    return new_g2o_lines