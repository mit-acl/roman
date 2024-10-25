import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import pickle
import tqdm
import yaml
import time
import cv2 as cv
import signal
import sys
import open3d as o3d
import logging
import os
from os.path import expandvars
from threading import Thread

from roman.map.run import ROMANMapRunner

from robotdatapy.data import ImgData

def main(args):
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)

    runner = ROMANMapRunner(params, verbose=True, viz_map=args.viz_map, 
                            viz_observations=args.viz_observations, save_viz=args.save_img_data)

    # Setup logging
    # TODO: add support for logfile
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(message)s', 
        datefmt='%m-%d %H:%M:%S', 
        # handlers=logging.StreamHandler()
    )

    print("Running segment tracking! Start time {:.1f}, end time {:.1f}".format(runner.t0, runner.tf))
    wc_t0 = time.time()

    vid = args.viz_map or args.viz_observations
    if vid:
        fc = cv.VideoWriter_fourcc(*"mp4v")
        video_file = os.path.expanduser(expandvars(args.output)) + ".mp4"
        fps = int(np.max([5., args.vid_rate*1/params['segment_tracking']['dt']]))
        if params['fastsam']['rotate_img'] not in ['CCW', 'CW']:
            width = runner.img_data.camera_params.width 
            height = runner.img_data.camera_params.height
        else:
            width = runner.img_data.camera_params.height
            height = runner.img_data.camera_params.width
        video = cv.VideoWriter(video_file, fc, fps, 
                               (width*(2 if (args.viz_map and args.viz_observations) else 1), height))

    for t in runner.times():
        img_t = runner.update(t)
        if vid and img_t is not None:
            video.write(img_t)
            
    if vid:
        video.release()
        cv.destroyAllWindows()

    print(f"Segment tracking took {time.time() - wc_t0:.2f} seconds")
    print(f"Run duration was {runner.tf - runner.t0:.2f} seconds")
    print(f"Compute per second: {(time.time() - wc_t0) / (runner.tf - runner.t0):.2f}")

    print(f"Number of poses: {len(runner.poses_flu_history)}.")

    if args.output:
        pkl_path = os.path.expanduser(expandvars(args.output)) + ".pkl"
        pkl_file = open(pkl_path, 'wb')
        runner.mapper.make_pickle_compatible()
        pickle.dump([runner.mapper, runner.poses_flu_history, runner.times_history], pkl_file, -1)
        logging.info(f"Saved tracker, poses_flu_history to file: {pkl_path}.")
        pkl_file.close()

        timing_file = os.path.expanduser(expandvars(args.output)) + ".time.txt"
        with open(timing_file, 'w') as f:
            f.write(f"dt: {params['segment_tracking']['dt']}\n\n")
            f.write(f"AVERAGE TIMES\n")
            f.write(f"fastsam: {np.mean(runner.processing_times.fastsam_times):.3f}\n")
            f.write(f"segment_track: {np.mean(runner.processing_times.map_times):.3f}\n")
            f.write(f"total: {np.mean(runner.processing_times.total_times):.3f}\n")
            f.write(f"TOTAL TIMES\n")
            f.write(f"total: {np.sum(runner.processing_times.total_times):.2f}\n")
    
    if args.save_img_data:
        img_data_path = os.path.expanduser(expandvars(args.output)) + ".img_data.zip"
        print(f"Saving visualization to {img_data_path}")
        img_data = ImgData(times=runner.times_history, imgs=runner.viz_imgs, data_type='raw')
        img_data.to_zip(img_data_path)

    if args.viz_open3d:
        poses_list = []
        pcd_list = []
        for Twb in runner.poses_flu_history:
            pose_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            pose_obj.transform(Twb)
            poses_list.append(pose_obj)
        for seg in runner.mapper.segments + runner.mapper.inactive_segments + runner.mapper.segment_graveyard:
            seg_points = seg.points
            if seg_points is not None:
                num_pts = seg_points.shape[0]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(seg_points)
                rand_color = np.random.uniform(0, 1, size=(1,3))
                rand_color = np.repeat(rand_color, num_pts, axis=0)
                pcd.colors = o3d.utility.Vector3dVector(rand_color)
                pcd_list.append(pcd)
        
        o3d.visualization.draw_geometries(poses_list + pcd_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, help='Path to params file', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to output file', required=False, default=None)
    parser.add_argument('-m', '--viz-map', action='store_true', help='Visualize map')
    parser.add_argument('-v', '--viz-observations', action='store_true', help='Visualize observations')
    parser.add_argument('-3', '--viz-open3d', action='store_true', help='Visualize in 3D')
    parser.add_argument('--vid-rate', type=float, help='Video playback rate', default=1.0)
    parser.add_argument('-d', '--save-img-data', action='store_true', help='Save video frames as ImgData class')
    args = parser.parse_args()

    main(args)