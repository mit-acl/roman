import numpy as np
import argparse
import gtsam
import yaml

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

def reformat_g2o_lines(lc_file, letter1, letter2, thresh=None, lc=False, self_lc=False):
    output_lines = []

    with open(lc_file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if line.startswith('#') or line == '':
                continue
            line = line.split()
            if len(line) != 31:
                assert False, f"Invalid line: {line}"
            assert line[0] == 'EDGE_SE3:QUAT', f"Invalid line: {line}"
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

def main(args):
    
    # get configruation from yaml file
    with open(args.yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    robot_letters = {r['robot']: r['letter'] for r in config['robots']}

    output_lines = []

    # add odometry lines
    for odom_config in config['odometry']:
        odom_file = odom_config['file']
        letter = robot_letters[odom_config['robot']]
        output_lines += reformat_g2o_lines(odom_file, letter, letter, args.thresh, lc=False)
        
    # add single robot loop closures
    for single_lc_config in config['single_lc']:
        lc_file = single_lc_config['file']
        letter = robot_letters[single_lc_config['robot']]
        output_lines += reformat_g2o_lines(lc_file, letter, letter, args.thresh, lc=True, self_lc=True)

    # add multi robot loop closures
    for multi_lc_config in config['multi_lc']:
        lc_file = multi_lc_config['file']
        letters = [robot_letters[multi_lc_config['robot1']], robot_letters[multi_lc_config['robot2']]]
        output_lines += reformat_g2o_lines(lc_file, letters[0], letters[1], args.thresh, lc=True)

    with open(args.output, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
        f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuse submap aligns and odometry into a single g2o file.')
    parser.add_argument('yaml', type=str, help='Input yaml file.')
    parser.add_argument('output', type=str, help='Output g2o file.')
    parser.add_argument('-t', '--thresh', type=int, default=None, 
                        help='Threshold on number of selected associations for submap alignment.')
    args = parser.parse_args()
    
    main(args)