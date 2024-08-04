import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main(args):
    with open(os.path.expanduser(args.input), 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    vertices = [line for line in lines if line[0] == 'VERTEX_SE3:QUAT']
    positions = np.array([[float(x) for x in line[2:5]] for line in vertices])
    print(positions[0,0])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(positions[:, 0], positions[:, 1], label='vertices')
    ax.legend()

    if args.output is not None:
        plt.savefig(os.path.expanduser(args.output))
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str, default=None)
    args = parser.parse_args()
    main(args)