import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import gtsam

def main(args):
    with open(os.path.expanduser(args.input), 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    robots = set([chr(gtsam.Symbol(int(line[1])).chr()) for line in lines if line[0] == 'VERTEX_SE3:QUAT'])
    vertices = [line for line in lines if line[0] == 'VERTEX_SE3:QUAT']
    positions = dict()
    for r in robots:
        positions[r] = np.array([[float(x) for x in line[2:5]] for line in vertices if chr(gtsam.Symbol(int(line[1])).chr()) == r])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for r in robots:
        ax.plot(positions[r][:, 1], -positions[r][:, 0], label=f'{r}', linewidth=3)
    ax.legend()
    ax.grid(True)

    if args.output is not None:
        plt.savefig(os.path.expanduser(args.output))
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()
    main(args)