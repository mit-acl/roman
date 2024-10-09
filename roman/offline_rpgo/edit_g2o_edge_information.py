import numpy as np
import os
import argparse
from typing import List

def std_dev_to_information_matrix(
    translation_std_dev: float,
    rotation_std_dev: float,
) -> np.ndarray:
    """Create information matrix from standard deviations.

    Args:
        translation_std_dev (float): translation standard deviation
        rotation_std_dev (float): rotation standard deviation

    Returns:
        np.ndarray: information matrix
    """
    
    I_t = 1 / (translation_std_dev**2)
    I_r = 1 / (rotation_std_dev**2)
    information_matrix = np.diag([I_t, I_t, I_t, I_r, I_r, I_r])
    return information_matrix
    

def std_dev_to_information_matrix_str(
    translation_std_dev: float = None, 
    rotation_std_dev: float = None,
    information_matrix: np.ndarray = None
) -> str:
    """
    Creates g2o information matrix string from standard deviations or information matrix.

    Args:
        translation_std_dev (float): tranlsation part standard deviation. Provide with rotation_std_dev.
        rotation_std_dev (float): rotation part standard deviation. Provide with translation_std_dev.
        information_matrix (np.ndarray): information matrix.

    Returns:
        str: g2o information matrix string
    """
    
    assert (translation_std_dev is not None and rotation_std_dev is not None) or \
        information_matrix is not None, "Provide either standard deviations or information matrix."
    assert (translation_std_dev is None and rotation_std_dev is None) or \
        information_matrix is None, "Provide either standard deviations or information matrix, not both."
    
    if information_matrix is None:
        information_matrix = std_dev_to_information_matrix(translation_std_dev, rotation_std_dev)
    
    ret = ""
    for i in range(6):
        for j in range(6):
            if j < i:
                continue
            ret += f"{information_matrix[i, j]} "
        ret += "\t"
    return ret

def edit_g2o_edge_information(
    g2o_lines: List[str], 
    translation_std_dev: float, 
    rotation_std_dev: float, 
    odometry: bool = False, 
    loop_closures: bool = False
) -> List[str]:
    """
    Change the information matrix in a g2o file

    Args:
        g2o_lines (List[str]): g2o file lines
        translation_std_dev (float): translation part standard deviation
        rotation_std_dev (float): rotation part standard deviation
        odometry (bool, optional): Change odometry edges. Defaults to False.
        loop_closures (bool, optional): Change loop closure edges. Defaults to False.
        
    Returns:
        List[str]: g2o file lines with updated information matrices
    """
    
    assert odometry or loop_closures, "Set either odometry or loop_closure to true."
    
    I = std_dev_to_information_matrix(translation_std_dev, rotation_std_dev)
    ret = []
    
    for i, line in enumerate(g2o_lines):
        line = line.strip()
        if "EDGE_SE3:QUAT" not in line:
            ret.append(line)
            continue
        
        line_data = line.split()
        v1 = int(line_data[1])
        v2 = int(line_data[2])
        
        if odometry and np.abs(v1 - v2) != 1:
            ret.append(line)
            continue
        elif loop_closures and np.abs(v1 - v2) == 1:
            ret.append(line)
            continue
        
        new_line = ' '.join(line_data[:10]) + '\t'
        new_line += std_dev_to_information_matrix_str(information_matrix=I)
        ret.append(new_line)
        
    return ret

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Edit information matrix in g2o file.')
    parser.add_argument('input', type=str, help='Input g2o file.')
    parser.add_argument('output', type=str, help='Output g2o file.')
    parser.add_argument('-l', '--loop-closures', type=float, nargs=2, default=(None, None),
                        help='Loop closure new translation (m) and rotation (deg) standard deviations.')
    parser.add_argument('-o', '--odometry', type=float, nargs=2, default=(None, None),
                        help='Odometry new translation (m) and rotation (deg) standard deviations.')
    args = parser.parse_args()
    
    with open(os.path.expanduser(args.input), 'r') as f:
        g2o_lines = f.readlines()
        
    if args.loop_closures != (None, None):
        g2o_lines = edit_g2o_edge_information(g2o_lines, args.loop_closures[0], np.deg2rad(args.loop_closures[1]), loop_closures=True)
    
    if args.odometry != (None, None):
        g2o_lines = edit_g2o_edge_information(g2o_lines, args.odometry[0], np.deg2rad(args.odometry[1]), odometry=True)
        
    with open(os.path.expanduser(args.output), 'w') as f:
        for line in g2o_lines:
            f.write(line + '\n')
        f.close()
        
        

