import numpy as np
import argparse
import os
import pickle

from robotdatapy import transform

def main(args):

    I_t = 1 / (args.t_std**2)
    I_r = 1 / (args.r_std**2)
    I = np.diag([I_t, I_t, I_t, I_r, I_r, I_r])
    
    with open(os.path.expanduser(args.input), 'rb') as f:
        pickle_data = pickle.load(f)
        if len(pickle_data) == 2:
            _, poses, = pickle_data
            times = None
        else:
            _, poses, times = pickle_data

    with open(os.path.expanduser(args.output), 'w') as f:
        for i, pose in enumerate(poses):
            t, q = transform.transform_to_xyz_quat(pose, separate=True)
            f.write(f"VERTEX_SE3:QUAT {i} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")

        for i in range(len(poses) - 1):
            T_w1 = poses[i]
            T_w2 = poses[i + 1]
            T_12 = np.linalg.inv(T_w1) @ T_w2
            t, q = transform.transform_to_xyz_quat(T_12, separate=True)
            f.write(f"EDGE_SE3:QUAT {i} {i + 1} \t\t")
            f.write(f"{t[0]} {t[1]} {t[2]} \t\t")
            f.write(f"{q[0]} {q[1]} {q[2]} {q[3]} \t\t")
            for i in range(6):
                for j in range(6):
                    if j < i:
                        continue
                    f.write(f"{I[i, j]} ")
                f.write("\t\t")
            f.write("\n")
    
        f.close()

    print(f"Saved g2o to {os.path.abspath(args.output)}")
    
    if args.output_time is None:
        return
    
    assert times is not None, "No time data found in pickle file."
    
    with open(os.path.expanduser(args.time_file), 'w') as f:
        for i, time in enumerate(times):
            f.write(f"{args.robot_id} {i} {int(time*1e9)} xxx\n")
        f.close()
        
    print(f"Saved time data to {os.path.abspath(args.time_file)}")


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SegTrack to g2o format.')
    parser.add_argument('input', type=str, help='Input SegTrack file.')
    parser.add_argument('output', type=str, help='Output g2o file.')
    parser.add_argument('-t', '--output-time', action='store_true', help='Output timing information.')
    parser.add_argument('-f', '--time-file', type=str, default=None, 
                        help='Time file. Saves with same name as g2o file with time.txt extension if this is not set.')
    parser.add_argument('-n', '--robot-id', type=int, default=0, help='Robot ID.')
    args = parser.parse_args()

    args.t_std = .01
    args.r_std = np.deg2rad(.02)
    
    if args.time_file is None and args.output_time:
        args.time_file = args.output.replace('.g2o', '_time.txt')
    
    main(args)