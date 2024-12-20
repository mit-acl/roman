import numpy as np
import argparse
import gtsam
import yaml
import os
from typing import List

def format_g2o_line(data):
    ret = data[0]
    ret += f'\t{data[1]} {data[2]} \t'
    ret += f'{data[3]} {data[4]} {data[5]} \t'
    ret += f'{data[6]} {data[7]} {data[8]} {data[9]} \t'
    ret += f'{data[10]} {data[11]} {data[12]} {data[13]} {data[14]} {data[15]} \t'
    ret += f'{data[16]} {data[17]} {data[18]} {data[19]} {data[20]} \t'
    ret += f'{data[21]} {data[22]} {data[23]} {data[24]} \t'
    ret += f'{data[25]} {data[26]} {data[27]} \t'
    ret += f'{data[28]} {data[29]} \t'
    ret += f'{data[30]}'
    return ret

def reformat_g2o_vertex_lines(file, letter):
    output_lines = []

    with open(os.path.expandvars(file), 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if line.startswith('#') or line == '':
                continue
            line = line.split()
            
            assert (line[0] == 'EDGE_SE3:QUAT' and len(line) == 31) or \
                    (line[0] == 'VERTEX_SE3:QUAT' and len(line) == 9), f"Invalid line: {line}"
            if line[0] == 'EDGE_SE3:QUAT':
                continue

            line[1] = ''.join(ch for ch in line[1] if ch.isdigit())
            line[1] = str(gtsam.symbol(letter, int(line[1])))
            output_lines.append(' '.join(line))
            
    return output_lines
            

def reformat_g2o_edge_lines(file, letter1, letter2, thresh=None, lc=False, self_lc=False):
    output_lines = []

    with open(os.path.expandvars(file), 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if line.startswith('#') or line == '':
                continue
            line = line.split()
            assert (line[0] == 'EDGE_SE3:QUAT' and len(line) == 31) or \
                    (line[0] == 'VERTEX_SE3:QUAT' and len(line) == 9), f"Invalid line: {line}"
                    
            if line[0] == 'VERTEX_SE3:QUAT':
                continue
            
            if self_lc: # make sure we only add self loop closures once
                if line[1] >= line[2]:
                    continue
            if lc: # filter out loop closures with less than a certain number of associations
                prev_line = lines[i - 1].strip()
                assert prev_line.startswith('# LC:'), "loop closure must be preceded by a comment"
                num_assoc = int(prev_line.split()[2])
                if thresh is not None and num_assoc < thresh:
                    continue

            line[1] = ''.join(ch for ch in line[1] if ch.isdigit())
            line[2] = ''.join(ch for ch in line[2] if ch.isdigit())
            line[1] = gtsam.symbol(letter1, int(line[1]))
            line[2] = gtsam.symbol(letter2, int(line[2]))
            
            output_lines.append(format_g2o_line(line))
    return output_lines

def create_config(robots, odometry_g2o_dir, submap_align_dir=None, align_file_name=None):
    """
    Creates config dict for g2o file fusion.

    Args:
        robots (List[str]): List of robot names used in the g2o files. Defaults to None.
        odometry_g2o_dir (str, optional): Odometry g2o file directory. Defaults to None.
        submap_align_dir (str, optional): Submap align results file directory. Defaults to None.
        align_file_name (str, optional): Name of files used in submap align. Defaults to None.
    """
    config = {}
    config['robots'] = []
    config['odometry'] = []
    config['single_lc'] = []
    config['multi_lc'] = []
    for i, robot in enumerate(robots):
        config['robots'].append({'robot': robot, 'letter': chr(ord('a') + i)})
        config['odometry'].append({'robot': robot, 'file': f'{odometry_g2o_dir}/{robot}.g2o'})
        if submap_align_dir is not None:
            config['single_lc'].append({'robot': robot, 'file': f'{submap_align_dir}/{robot}_{robot}/{align_file_name}.g2o'})
            for j, robot2 in enumerate(robots):
                if i >= j:
                    continue
                config['multi_lc'].append({'robot1': robot, 'robot2': robot2, 'file': f'{submap_align_dir}/{robot}_{robot2}/{align_file_name}.g2o'})
    return config

def g2o_file_fusion(
    config: dict,
    output: str,
    thresh: int = None
):
    """
    Fuses a series of single robot odometry g2o files and multi-robot/single-robot 
        loop closure g2o files into a single g2o file.
    Args:
        output (str): Output file path.
        thresh (int, optional): _description_. Defaults to None.
    """
    
    robot_letters = {r['robot']: r['letter'] for r in config['robots']}

    output_lines = []

    # add odometry lines
    for odom_config in config['odometry']:
        odom_file = odom_config['file']
        letter = robot_letters[odom_config['robot']]
        output_lines += reformat_g2o_edge_lines(odom_file, letter, letter, thresh, lc=False)
        output_lines += reformat_g2o_vertex_lines(odom_file, letter)
        
        
    # add single robot loop closures
    for single_lc_config in config['single_lc']:
        lc_file = single_lc_config['file']
        letter = robot_letters[single_lc_config['robot']]
        output_lines += reformat_g2o_edge_lines(lc_file, letter, letter, thresh, lc=True, self_lc=True)

    # add multi robot loop closures
    for multi_lc_config in config['multi_lc']:
        lc_file = multi_lc_config['file']
        letters = [robot_letters[multi_lc_config['robot1']], robot_letters[multi_lc_config['robot2']]]
        output_lines += reformat_g2o_edge_lines(lc_file, letters[0], letters[1], thresh, lc=True)

    with open(output, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
        f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuse submap aligns and odometry into a single g2o file.')
    parser.add_argument('--yaml', '-y', type=str, help='Input yaml file.')
    parser.add_argument('--output', '-o', type=str, help='Output g2o file.')
    parser.add_argument('--robots', '-r', type=str, nargs='+', help='Robots to include in the g2o file.')
    parser.add_argument('--odometry-g2o', '-g', type=str, help='odometry g2o directory path.')
    parser.add_argument('--submap-align', '-a', type=str, help='submap align directory path.')
    parser.add_argument('--align-file-name', '-n', type=str, help='submap align file name.')
    parser.add_argument('-t', '--thresh', type=int, default=None, 
                        help='Threshold on number of selected associations for submap alignment.')
    args = parser.parse_args()
    
    if args.yaml is not None:
        # get configruation from yaml file
        with open(args.yaml, 'r') as f:
            config = yaml.safe_load(f)
    else:
        assert args.robots is not None and args.odometry_g2o is not None, \
            "Must provide either a yaml file or robots and odometry_g2o arguments."
        config = create_config(args.robots, args.odometry_g2o, args.submap_align, args.align_file_name)
    
    g2o_file_fusion(
        config,
        args.output,
        args.thresh
    )