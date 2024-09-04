import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import gtsam

from robotdatapy.transform import xyz_quat_to_transform, transform_to_xyzrpy

KMD_ROBOTS = ['acl_jackal', 'acl_jackal2', 'sparkal1', 'sparkal2', 'hathor', 'thoth', 'apis', 'sobek']

def main(args):
    colors = {
        'a': 'tab:blue',
        'b': 'tab:orange',
        'c': 'tab:green',
        'd': 'tab:pink',
        'e': 'tab:purple',
        'f': 'tab:brown',
        'g': 'tab:red',
        'h': 'tab:gray',
    }
    if args.robot_letters is None:
        names = {chr(97 + i): args.robots[i] for i in range(len(args.robots))}
    else:
        names = {args.robot_letters[i]: args.robots[i] for i in range(len(args.robots))}
    # names = {
    #     'a': 'acl_jackal',
    #     'b': 'acl_jackal2',
    #     'c': 'sparkal1',
    #     'd': 'sparkal2',
    #     'e': 'hathor',
    #     'f': 'thoth',
    #     'g': 'apis',
    #     'h': 'sobek'
    # }
    linewidth = 3
    linewidth_lc = 2.0

    with open(os.path.expanduser(args.input), 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    robots = set([chr(gtsam.Symbol(int(line[1])).chr()) for line in lines if line[0] == 'VERTEX_SE3:QUAT'])
    vertices = [line for line in lines if line[0] == 'VERTEX_SE3:QUAT']
    positions = dict()
    pose_by_index = {int(line[1]): xyz_quat_to_transform(np.array([float(x) for x in line[2:5]]), 
                                  np.array([float(x) for x in line[5:9]])) for line in vertices}
    for r in robots:
        positions[r] = np.array([[float(x) for x in line[2:5]] for line in vertices if chr(gtsam.Symbol(int(line[1])).chr()) == r])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for r in robots:
        ax.plot(positions[r][:, 0], positions[r][:, 1], label=f'{names[r]}', linewidth=linewidth, color=colors[r])
    ax.legend()

    if args.loop_closures:
        edges = [line for line in lines if line[0] == 'EDGE_SE3:QUAT']
        for edge in edges:
            v1 = int(edge[1])
            v2 = int(edge[2])
            r1 = chr(gtsam.Symbol(v1).chr())
            r2 = chr(gtsam.Symbol(v2).chr())
            if args.no_self and r1 == r2:
                continue
            if np.abs(v2 - v1) == 1:
                continue
            T_12_lc = xyz_quat_to_transform(np.array([float(x) for x in edge[3:6]]), 
                                      np.array([float(x) for x in edge[6:10]]))
            T_w1 = pose_by_index[v1]
            T_w2 = pose_by_index[v2]
            T_12 = np.linalg.inv(T_w1) @ T_w2
            p1 = pose_by_index[v1][:3, 3]
            p2 = pose_by_index[v2][:3, 3]
            # print(np.linalg.norm(T_12 @ np.linalg.inv(T_12_lc) - np.eye(4), ord='fro'))
            # print(T_12_lc)
            # print(T_12)
            T_err = T_12 @ np.linalg.inv(T_12_lc) 
            xyz_rpy_err = transform_to_xyzrpy(T_err).reshape((6,1))
            information_mat = np.eye(6)
            information_mat[0,:] = [float(x) for x in edge[10:16]]
            information_mat[1,1:] = [float(x) for x in edge[16:21]]
            information_mat[2,2:] = [float(x) for x in edge[21:25]]
            information_mat[3,3:] = [float(x) for x in edge[25:28]]
            information_mat[4,4:] = [float(x) for x in edge[28:30]]
            information_mat[5,5] = float(edge[30])
            mahalanobis = np.sqrt(xyz_rpy_err.T @ information_mat @ xyz_rpy_err)
            inlier = mahalanobis < 2.0
            # if inlier:
            #     print(np.linalg.norm(T_12_lc[:3,3]))
            
            # inlier = np.linalg.norm(T_12 @ np.linalg.inv(T_12_lc) - np.eye(4), ord='fro') < 1.0
            if args.inliers_only and not inlier:
                continue
            if args.outliers_only and inlier:
                continue
            color = 'lawngreen' if inlier else 'red'
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color, linewidth=linewidth_lc, alpha=1)

    ax.grid(True)

    if args.output is not None:
        plt.savefig(os.path.expanduser(args.output), transparent=False)
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-l', '--loop-closures', action='store_true',
                        help='Plot loop closures')
    parser.add_argument('-i', '--inliers-only', action='store_true',
                        help='Plot only inliers')
    parser.add_argument('-j', '--outliers-only', action='store_true',
                        help='Plot only outliers')
    parser.add_argument('-r', '--robots', type=str, nargs="+", default=KMD_ROBOTS,)
    parser.add_argument('--robot-letters', type=str, nargs="+", default=None,
                        help='Specify the letters to use for each robot')
    parser.add_argument('--no-self', action='store_true',
                        help='Do not plot self loop closures')
    args = parser.parse_args()

    assert not (args.inliers_only and args.outliers_only), "Cannot specify both inliers-only and outliers-only"

    main(args)