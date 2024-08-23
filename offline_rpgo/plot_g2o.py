import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import gtsam

def main(args):
    colors = {
        'a': 'tab:blue',
        'b': 'tab:orange',
        'c': 'tab:green',
        'd': 'tab:pink',
        'e': 'tab:purple',
        'f': 'tab:brown',
        'g': 'tab:pink',
        'h': 'tab:gray',
    }
    linewidth = 3

    with open(os.path.expanduser(args.input), 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    robots = set([chr(gtsam.Symbol(int(line[1])).chr()) for line in lines if line[0] == 'VERTEX_SE3:QUAT'])
    vertices = [line for line in lines if line[0] == 'VERTEX_SE3:QUAT']
    positions = dict()
    positions_by_idx = {int(line[1]): np.array([float(x) for x in line[2:5]]) for line in vertices}
    for r in robots:
        positions[r] = np.array([[float(x) for x in line[2:5]] for line in vertices if chr(gtsam.Symbol(int(line[1])).chr()) == r])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for r in robots:
        ax.plot(positions[r][:, 0], positions[r][:, 1], label=f'{r}', linewidth=linewidth, color=colors[r])
    ax.legend()

    if args.loop_closures:
        edges = [line for line in lines if line[0] == 'EDGE_SE3:QUAT']
        for edge in edges:
            v1 = int(edge[1])
            v2 = int(edge[2])
            if np.abs(v2 - v1) == 1:
                continue
            p1 = positions_by_idx[v1]
            p2 = positions_by_idx[v2]
            d = np.linalg.norm(p2 - p1)
            inlier = d < 10.0
            if args.inliers_only and not inlier:
                continue
            if args.outliers_only and inlier:
                continue
            color = 'lawngreen' if inlier else 'red'
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color, linewidth=linewidth, alpha=1)

    ax.grid(True)

    if args.output is not None:
        plt.savefig(os.path.expanduser(args.output))
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
    args = parser.parse_args()

    assert not (args.inliers_only and args.outliers_only), "Cannot specify both inliers-only and outliers-only"

    main(args)