import argparse
import numpy as np
import pickle
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import yaml

from robotdatapy.data import PoseData
from roman.params.data_params import DataParams
from roman.params.submap_align_params import SubmapAlignParams
from roman.map.map import submaps_from_roman_map, ROMANMap, SubmapParams, Submap, load_roman_map

import colorsys
import random


SCALE_FACTOR = 10
COLOR_QUEUE_LEN = 3
COLOR_QUEUE_INVALID_DIST = np.array([0.05, 0.1, 0.15])


def generate_bright_color_palette(num_colors):
    colors = []
    for i in range(num_colors):
        h = i / num_colors  # Evenly distribute hues
        s = random.uniform(0.5, 1.0)  # Saturation: keep it high
        v = random.uniform(0.5, 1.0)  # Value/Brightness: also keep it high
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append(f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})')
    return colors

def random_bright_color(last_colors=None):
    last_hs = []
    for last_color in last_colors:
        last_h, _, _ = colorsys.rgb_to_hsv(*last_color)
        last_hs.append(last_h)
    while True:
        h = random.random()
        diff = True
        for i in range(len(last_hs)):
            if np.abs(last_hs[i] - h) < COLOR_QUEUE_INVALID_DIST[i]:
                diff = False
                break
        if diff: break

    s = random.uniform(0.5, 1.0)  # Saturation: keep it high
    v = random.uniform(0.5, 1.0)  # Value/Brightness: also keep it high
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})', (r, g, b)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--params', default=None, type=str, required=True)
parser.add_argument('-o', '--output-dir', type=str, required=True, default=None)
parser.add_argument('--runs', '-r', type=str, nargs='+', default=None)
parser.add_argument('--visualize_together', '-t', action='store_true')

args = parser.parse_args()

viz_together = args.visualize_together
params_dir = args.params

submap_align_params_path = os.path.join(args.params, f"submap_align.yaml")
submap_align_params = SubmapAlignParams.from_yaml(submap_align_params_path) \
    if os.path.exists(submap_align_params_path) else SubmapAlignParams()
data_params = DataParams.from_yaml(os.path.join(args.params, "data.yaml"))
if args.runs is not None:
    data_params.runs = args.runs

if os.path.exists(os.path.join(params_dir, "gt_pose.yaml")):
    has_gt = True
    gt_files = [os.path.join(params_dir, "gt_pose.yaml") for _ in range(len(data_params.runs))]
else:
    has_gt = False
    gt_files = [None for _ in range(len(data_params.runs))]


run_submaps = []
run_gt_pose_data = []

for i in range(len(data_params.runs)):
    run = data_params.runs[i]
    map_file = os.path.join(args.output_dir, "map", f"{run}.pkl")
    gt_file = gt_files[i]

    gt_pose_data = None
    # load gt pose data
    if gt_file is not None:
        if data_params.run_env is not None:
            os.environ[data_params.run_env] = run
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
        
    submap_params = SubmapParams.from_submap_align_params(submap_align_params)
    submap_params.use_minimal_data = True
    roman_map = load_roman_map(map_file)
    submaps = submaps_from_roman_map(roman_map, submap_params, gt_pose_data)

    run_submaps.append(submaps)
    run_gt_pose_data.append(gt_pose_data)


if viz_together:
    fig = go.Figure()
    title = f"{args.output_dir} (gt={has_gt})"
    run_colors = generate_bright_color_palette(len(data_params.runs))

    # Add legend for runs
    for i, run_color in enumerate(run_colors):
        fig.add_trace(go.Scatter(
            x=[None],  # Dummy point for legend
            y=[None],
            mode='markers',
            marker=dict(color=run_color),
            name=f'{data_params.runs[i]} ({len(run_submaps[i])} submaps)',
            showlegend=True
        ))

for i in range(len(data_params.runs)):
    run = data_params.runs[i]
    submaps = run_submaps[i]
    gt_pose_data = run_gt_pose_data[i]

    if not viz_together:
        fig = go.Figure()
        title = f"[{args.output_dir}] {run} ({len(submaps)} submaps, gt={has_gt})"
    else:
        run_color = run_colors[i]

    last_submap_center = None
    last_color, last_color_raw_queue = None, []

    for submap in submaps:
        segment_points = submap.segments_as_global_points
        color, color_raw = random_bright_color(last_color_raw_queue)

        # Plot segment points
        if segment_points is not None:
            fig.add_trace(go.Scatter(
                x=segment_points[:, 0],
                y=segment_points[:, 1],
                mode='markers',
                marker=dict(color=color, size=SCALE_FACTOR),
                name=f'{run + " " if viz_together else ""}({submap.id})',
                showlegend=False
            ))

        # Submap center
        submap_center = submap.position_gt if submap.has_gt else submap.position
        fig.add_trace(go.Scatter(
            x=[submap_center[0]],
            y=[submap_center[1]],
            mode='markers',
            marker=dict(
                color=color,
                size=SCALE_FACTOR * 3,
                line=dict(
                    color='black',
                    width=SCALE_FACTOR/4
                )
            ),
            name=f'{run + " " if viz_together else ""}({submap.id}) center',
            showlegend=False
        ))

        # Connection to previous submap center
        if last_submap_center is not None:
            midpoint = (last_submap_center + submap_center) / 2

            # Line to midpoint (from last to midpoint)
            fig.add_trace(go.Scatter(
                x=[last_submap_center[0], midpoint[0]],
                y=[last_submap_center[1], midpoint[1]],
                mode='lines',
                line=dict(color=run_color if viz_together else last_color, width=SCALE_FACTOR, dash='solid'),
                opacity=0.5,
                showlegend=False
            ))

            # Line to current (from midpoint to current)
            fig.add_trace(go.Scatter(
                x=[midpoint[0], submap_center[0]],
                y=[midpoint[1], submap_center[1]],
                mode='lines',
                line=dict(color=run_color if viz_together else color, width=SCALE_FACTOR, dash='solid'),
                opacity=0.5,
                showlegend=False
            ))

        last_submap_center = submap_center
        last_color = color
        last_color_raw_queue.append(color_raw)
        if len(last_color_raw_queue) > COLOR_QUEUE_LEN:
            last_color_raw_queue.pop(0)

    if not viz_together or i == len(data_params.runs) - 1:

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='X',
            yaxis_title='Y',
            template='plotly_white',
            showlegend=viz_together,
            hovermode='closest',
            legend=dict(
                font=dict(size=20)
            ),
            title_font=dict(size=24),
            hoverlabel=dict(
                font_size=12,
                namelength=-1  # Extend popup length to show full text
            )
        )

        fig.update_layout(dragmode='pan')
        fig.show(config={"scrollZoom": True})


