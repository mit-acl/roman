import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import yaml

from robotdatapy.data import PoseData
from roman.params.data_params import DataParams
from roman.map.map import submaps_from_roman_map, ROMANMap, SubmapParams, Submap

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--params', default=None, type=str, required=True)
parser.add_argument('-o', '--output-dir', type=str, required=True, default=None)
parser.add_argument('--runs', '-r', type=str, nargs='+', default=None)
args = parser.parse_args()

params_dir = args.params

data_params = DataParams.from_yaml(os.path.join(args.params, "data.yaml"))
if args.runs is not None:
    data_params.runs = args.runs

if os.path.exists(os.path.join(params_dir, "gt_pose.yaml")):
    has_gt = True
    gt_files = [os.path.join(params_dir, "gt_pose.yaml") for _ in range(len(data_params.runs))]
else:
    has_gt = False
    gt_files = [None for _ in range(len(data_params.runs))]



for i in range(len(data_params.runs)):
    map_file = os.path.join(args.output_dir, "map", f"{data_params.runs[i]}.pkl")
    gt_file = gt_files[i]

    gt_pose_data = None
    # load gt pose data
    if gt_file is not None:
        if data_params.run_env is not None:
            os.environ[data_params.run_env] = data_params.runs[i]
        with open(os.path.expanduser(gt_file), 'r') as f:
            gt_pose_args = yaml.safe_load(f)
        if gt_pose_args['type'] == 'bag':
            gt_pose_data = PoseData.from_bag(**{k: v for k, v in gt_pose_args.items() if k != 'type'})
        elif gt_pose_args['type'] == 'csv':
            gt_pose_data = PoseData.from_csv(**{k: v for k, v in gt_pose_args.items() if k != 'type'})
        elif gt_pose_args['type'] == 'bag_tf':
            gt_pose_data = PoseData.from_bag_tf(**{k: v for k, v in gt_pose_args.items() if k != 'type'})
        else:
            raise ValueError("Invalid pose data type")
