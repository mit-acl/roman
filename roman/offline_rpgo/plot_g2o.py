import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import gtsam
from dataclasses import dataclass, field
from collections import defaultdict

from robotdatapy.transform import xyz_quat_to_transform, transform_to_xyzrpy
from robotdatapy import transform

KMD_ROBOTS = ['acl_jackal', 'acl_jackal2', 'sparkal1', 'sparkal2', 'hathor', 'thoth', 'apis', 'sobek']
DEFAULT_LC_COLORS = {
    'inlier': {
        'inter': 'lawngreen',
        'intra': 'xkcd:aqua blue',
    },
    'outlier': {
        'inter': 'red',
        'intra': 'red',
    }
}
DEFAULT_TRAJECTORY_COLORS = {
        'a': 'tab:blue',
        'b': 'tab:orange',
        'c': 'tab:green',
        'd': 'tab:pink',
        'e': 'tab:purple',
        'f': 'tab:brown',
        'g': 'tab:red',
        'h': 'tab:gray',
    }

@dataclass
class G2OPlotParams:
    inter: bool = True
    intra: bool = True
    inliers: bool = True
    outliers: bool = False
    colors: dict = field(default_factory=lambda: DEFAULT_LC_COLORS)
    lc_alpha: float = .9
    odom_alpha: float = 1.0
    odom_linewidth: float = 3.0
    lc_linewidth: float = 2.0
    legend: bool = True
    unconnected_robot_transform: dict = None
    axes: list = field(default_factory=lambda: [0, 1])

def plot_g2o(
    g2o_path,
    g2o_symbol_to_name,
    g2o_symbol_to_color,
    params = G2OPlotParams(),
    ax=None,
    map_transform=None
):

    if ax is None:
        fig, ax = plt.subplots()

    with open(os.path.expanduser(g2o_path), 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    robots = set([chr(gtsam.Symbol(int(line[1])).chr()) for line in lines if line[0] == 'VERTEX_SE3:QUAT'])
    vertices = [line for line in lines if line[0] == 'VERTEX_SE3:QUAT']
    positions = dict()
    pose_by_index = {int(line[1]): xyz_quat_to_transform(np.array([float(x) for x in line[2:5]]), 
                                  np.array([float(x) for x in line[5:9]])) for line in vertices}
    for r in robots:
        positions[r] = np.array([[float(x) for x in line[2:5]] for line in vertices if chr(gtsam.Symbol(int(line[1])).chr()) == r])
        if params.unconnected_robot_transform is not None and r in params.unconnected_robot_transform:
            positions[r] = transform.transform(params.unconnected_robot_transform[r], positions[r])
        if map_transform is not None:
            positions[r] = transform.transform(map_transform, positions[r])

    ax.set_aspect('equal')
    for r in robots:
        ax.plot(positions[r][:, params.axes[0]], positions[r][:, params.axes[1]], label=f'{g2o_symbol_to_name[r]}', linewidth=params.odom_linewidth, color=g2o_symbol_to_color[r], alpha=params.odom_alpha)
    if params.legend:
        ax.legend()

    if params.inter or params.intra:
        edges = [line for line in lines if line[0] == 'EDGE_SE3:QUAT']
        for edge in edges:
            v1 = int(edge[1])
            v2 = int(edge[2])
            r1 = chr(gtsam.Symbol(v1).chr())
            r2 = chr(gtsam.Symbol(v2).chr())
            if params.unconnected_robot_transform is not None:
                assert not (r1 in params.unconnected_robot_transform or r2 in params.unconnected_robot_transform) \
                    or (r1 == r2), "Cannot plot loop closures between unconnected robots"
            if not params.intra and r1 == r2: # Skip intra-robot loop closures
                continue
            if np.abs(v2 - v1) == 1:
                continue

            # determine if inlier or outlier
            T_12_lc = xyz_quat_to_transform(np.array([float(x) for x in edge[3:6]]), 
                                      np.array([float(x) for x in edge[6:10]]))
            T_w1 = pose_by_index[v1]
            T_w2 = pose_by_index[v2]
            T_12 = np.linalg.inv(T_w1) @ T_w2
            p1 = pose_by_index[v1][:3, 3]
            p2 = pose_by_index[v2][:3, 3]
            if params.unconnected_robot_transform is not None and r1 in params.unconnected_robot_transform:
                p1 = transform.transform(params.unconnected_robot_transform[r1], p1)
                p2 = transform.transform(params.unconnected_robot_transform[r2], p2)
            if map_transform is not None:
                p1 = transform.transform(map_transform, p1)
                p2 = transform.transform(map_transform, p2)
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

            if not params.outliers and not inlier:
                continue
            if not params.inliers and inlier:
                continue

            color = params.colors['inlier' if inlier else 'outlier']['inter' if r1 != r2 else 'intra']
            ax.plot([p1[params.axes[0]], p2[params.axes[0]]], [p1[params.axes[1]], p2[params.axes[1]]], color, linewidth=params.lc_linewidth, alpha=params.lc_alpha)

    ax.grid(True)

def main(args):
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
    fig, ax = plt.subplots()

    params = G2OPlotParams(
        inter=args.loop_closures,
        intra=args.loop_closures and not args.no_self,
        inliers=args.loop_closures and not args.outliers_only,
        outliers=args.loop_closures and not args.inliers_only,
        colors=DEFAULT_LC_COLORS
    )

    plot_g2o(
        g2o_path=args.input,
        g2o_symbol_to_name=names,
        g2o_symbol_to_color=DEFAULT_TRAJECTORY_COLORS,
        params=params,
        ax=ax
    )
    

    if args.output is not None:
        plt.savefig(os.path.expanduser(args.output), transparent=True)
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