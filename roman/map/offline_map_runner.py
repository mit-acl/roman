###########################################################
#
# mapping.py
#
# Demo code for running ROMAN mapping
#
# Authors: Mason Peterson, Yulun Tian, Lucas Jia, Qingyuan Li
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import pickle
import time
import cv2 as cv
import open3d as o3d
import logging
import os
from os.path import expandvars
from threading import Thread
from dataclasses import dataclass

from roman.map.run import ROMANMapRunner
from roman.params.data_params import DataParams
from roman.params.mapper_params import MapperParams
from roman.params.fastsam_params import FastSAMParams
from roman.utils import expandvars_recursive
from roman.map.merge_maps import merge_maps

from robotdatapy.data import ImgData

@dataclass
class VisualizationParams:
    viz_map: bool = False
    viz_observations: bool = False
    viz_3d: bool = False
    vid_rate: float = 1.0
    save_img_data: bool = False


def extract_params(data_params_path, fastsam_params_path, mapper_params_path, run_name=None):
    assert os.path.exists(data_params_path), "Data params file does not exist."
    data_params = DataParams.from_yaml(data_params_path, run=run_name)
        
    if os.path.exists(fastsam_params_path):
        fastsam_params = FastSAMParams.from_yaml(fastsam_params_path, run=run_name)
    else:
        fastsam_params = FastSAMParams()

    if os.path.exists(mapper_params_path):
        mapper_params = MapperParams.from_yaml(mapper_params_path, run=run_name)
    else:
        mapper_params = MapperParams()
    
    return data_params, fastsam_params, mapper_params

def run(
    data_params: DataParams, 
    fastsam_params: FastSAMParams, 
    mapper_params: MapperParams,
    output_path: str,
    viz_params: VisualizationParams = VisualizationParams(),
    verbose: bool = False
):
    
    runner = ROMANMapRunner(data_params=data_params, 
                            fastsam_params=fastsam_params, 
                            mapper_params=mapper_params, 
                            verbose=True, viz_map=viz_params.viz_map, 
                            viz_observations=viz_params.viz_observations, 
                            viz_3d=viz_params.viz_3d,
                            save_viz=viz_params.save_img_data)

    # Setup logging
    # TODO: add support for logfile
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(message)s', 
        datefmt='%m-%d %H:%M:%S', 
        # handlers=logging.StreamHandler()
    )

    if verbose: print("\n---\nRunning segment tracking! Start time {:.1f}, end time {:.1f}\n---\n".format(runner.t0, runner.tf))
    wc_t0 = time.time()

    vid = viz_params.viz_map or viz_params.viz_observations
    if vid:
        fc = cv.VideoWriter_fourcc(*"mp4v")
        video_file = os.path.expanduser(expandvars(output_path)) + ".mp4"
        fps = int(np.max([5., viz_params.vid_rate*1/data_params.dt]))
        if fastsam_params.rotate_img not in ['CCW', 'CW']:
            width = runner.img_data.camera_params.width 
            height = runner.img_data.camera_params.height
        else:
            width = runner.img_data.camera_params.height
            height = runner.img_data.camera_params.width
        num_panes = 0
        if viz_params.viz_map:
            num_panes += 1
        if viz_params.viz_observations:
            num_panes += 1
        if viz_params.viz_3d:
            num_panes += 1
        video = cv.VideoWriter(video_file, fc, fps, 
                               (width*num_panes, height))

    for t in runner.times():
        img_t = runner.update(t)
        if vid and img_t is not None:
            video.write(img_t)
            
    if vid:
        video.release()

    if verbose: 
        print(f"Segment tracking took {time.time() - wc_t0:.2f} seconds")
        print(f"Run duration was {runner.tf - runner.t0:.2f} seconds")
        print(f"Compute per second: {(time.time() - wc_t0) / (runner.tf - runner.t0):.2f}")

        print(f"Number of poses: {len(runner.mapper.poses_flu_history)}.")

    # Output results
    pkl_path = os.path.expanduser(expandvars(output_path)) + ".pkl"
    pkl_file = open(pkl_path, 'wb')
    pickle.dump(runner.mapper.get_roman_map(), pkl_file, -1)
    logging.info(f"Saved tracker, poses_flu_history to file: {pkl_path}.")
    pkl_file.close()

    timing_file = os.path.expanduser(expandvars(output_path)) + ".time.txt"
    with open(timing_file, 'w') as f:
        f.write(f"dt: {data_params.dt}\n\n")
        f.write(f"AVERAGE TIMES\n")
        f.write(f"fastsam: {np.mean(runner.processing_times.fastsam_times):.3f}\n")
        f.write(f"segment_track: {np.mean(runner.processing_times.map_times):.3f}\n")
        f.write(f"total: {np.mean(runner.processing_times.total_times):.3f}\n")
        f.write(f"TOTAL TIMES\n")
        f.write(f"total: {np.sum(runner.processing_times.total_times):.2f}\n")
    
    if viz_params.save_img_data:
        img_data_path = os.path.expanduser(expandvars(output_path)) + ".img_data.npz"
        if verbose: print(f"Saving visualization to {img_data_path}")
        img_data = ImgData(times=runner.mapper.times_history, imgs=runner.viz_imgs, data_type='raw')
        img_data.to_npz(img_data_path)
    
    del runner
    return

def mapping(
    params_path: str,
    output_path: str,
    run_name: str = None,
    max_time: float = None,
    viz_params: VisualizationParams = VisualizationParams(),
    verbose: bool = False
):
    data_params_path = expandvars_recursive(f"{params_path}/data.yaml")
    mapper_params_path = expandvars_recursive(f"{params_path}/mapper.yaml")
    fastsam_params_path = expandvars_recursive(f"{params_path}/fastsam.yaml")
        
    data_params = DataParams.from_yaml(data_params_path, run=run_name)
    if max_time is not None: data_params.max_time = max_time

    if verbose: print(f"Map iteration max time: {data_params.max_time}")

    if data_params.max_time is not None:
        # store original (complete) time range
        if data_params.time_params is not None:
            start_time = data_params.time_params['t0']
            end_time = data_params.time_params['tf']
            relative = data_params.time_params['relative']
        else: # default: run until no data
            start_time, end_time = data_params.data_t_range
            relative = False

        if verbose: 
            if relative:
                sr, er, snr, enr = start_time, end_time, data_params.data_t0 + start_time, data_params.data_t0 + end_time
            else:
                sr, er, snr, enr = start_time - data_params.data_t0, end_time - data_params.data_t0, start_time, end_time
            print(f"Mapping iterations will cover time [{sr}, {er}] = [{snr}, {enr}]")

        mapping_iter = 0
        
        # while iteration start time is less than original end time
        # (add dt/2 in case of floating point error)
        while start_time + data_params.max_time * mapping_iter + data_params.dt / 2.0 < end_time:

            _, fastsam_params, mapper_params = \
                extract_params(data_params_path, fastsam_params_path, mapper_params_path, run_name=run_name)

            # time range for this iteration, bounded by end_time
            data_params.time_params = {
                't0': start_time + data_params.max_time * mapping_iter, 
                'tf': min(start_time + data_params.max_time * (mapping_iter + 1), end_time),
                'relative': relative}
            
            if verbose: print(f"\n---\nRunning mapping iteration {mapping_iter}\n---\n")
            run(data_params, fastsam_params, mapper_params, 
                output_path=f"{output_path}_{mapping_iter}", viz_params=viz_params, verbose=verbose)
            
            mapping_iter += 1

        merge_maps(demo_output_files, f"{output_path}.pkl")
        # merge files
        demo_output_files = [f"{output_path}_{mi}.pkl" for mi in range(mapping_iter)]
    
    else:
        if verbose: print(f"\n---\nRunning mapping\n---\n")
        data_params, fastsam_params, mapper_params = \
            extract_params(data_params_path, fastsam_params_path, mapper_params_path, run_name=run_name)
        run(data_params, fastsam_params, mapper_params, output_path, viz_params, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, help='Path to params file', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to output file', required=True)
    parser.add_argument('--max-time', type=float, default=None)
    parser.add_argument('-m', '--viz-map', action='store_true', help='Visualize map')
    parser.add_argument('-v', '--viz-observations', action='store_true', help='Visualize observations')
    parser.add_argument('-3', '--viz-3d', action='store_true', help='Visualize in 3D')
    parser.add_argument('--vid-rate', type=float, help='Video playback rate', default=1.0)
    parser.add_argument('-d', '--save-img-data', action='store_true', help='Save video frames as ImgData class')
    parser.add_argument('-r', '--run', type=str, help='Robot run', default=None)
    args = parser.parse_args()

    viz_params = VisualizationParams(
        viz_map=args.viz_map,
        viz_observations=args.viz_observations,
        viz_3d=args.viz_3d,
        vid_rate=args.vid_rate,
        save_img_data=args.save_img_data
    )

    mapping(
        params_path=args.params,
        output_path=args.output,
        run_name=args.run,
        max_time=args.max_time,
        viz_params=viz_params
    )