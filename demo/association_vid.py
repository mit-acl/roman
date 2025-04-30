import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pickle
import cv2 as cv
from dataclasses import dataclass
import os

from roman.map.map import Submap, SubmapParams, submaps_from_roman_map, load_roman_map
from roman.align.results import SubmapAlignResults, plot_align_results, submaps_from_align_results
from roman.params.data_params import DataParams
from roman.viz import visualize_segment_on_img

@dataclass
class VideoParams:

    fps: int = 10
    min_segment_dist: float = 15.0 # segment has to be within this distance to be drawn

def get_path(path_str) -> Path:
    path = Path(path_str).expanduser()
    return path

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--runs', '-r', type=str, nargs=2, required=True)
    parser.add_argument('--idx', '-i', type=int, nargs=2, default=None)

    args = parser.parse_args()
    results_dir = get_path(args.results_dir)
    results_align_file = results_dir / 'align' / f'{args.runs[0]}_{args.runs[1]}' / 'align.pkl'
    params_dir = results_dir / 'params'
    output_path = get_path(args.output_path)
    vid_params = VideoParams()


    print('Loading align results...')
    pkl_file = results_align_file.open('rb')
    results: SubmapAlignResults
    results = pickle.load(pkl_file)
    pkl_file.close()

    submaps = submaps_from_align_results(results)
    
    print(f'Loaded {len(submaps[0])} and {len(submaps[1])} submaps.')

    # get submap pair
    if args.idx is not None:
        idxs = args.idx
    else:
        plot_align_results(results, dpi=100)
        plt.show()

        idx_str = input("Please input two indices, separated by a space: \n")
        idxs = [int(idx) for idx in idx_str.split()]

    # extract time range
    submap_pair = [submaps[i][idxs[i]] for i in range(2)]
    time_ranges = []
    for i in range(2):
        t0 = np.min([seg.first_seen for seg in submap_pair[i].segments])
        tf = np.max([seg.last_seen for seg in submap_pair[i].segments])
        time_ranges.append(np.array([t0, tf]))
    
    # make time ranges some duration
    min_time_range = np.min([time_ranges[i][1] - time_ranges[i][0] for i in range(2)])
    for i in range(2):
        time_range_i = time_ranges[i][1] - time_ranges[i][0]
        if time_range_i >= min_time_range:
            time_range_diff = time_range_i - min_time_range
            time_ranges[i][0] += time_range_diff/2
            time_ranges[i][1] -= time_range_diff/2

    # Load data
    data_params = [DataParams.from_yaml(str(params_dir / "data.yaml"), run=run) for run in args.runs]
    for i in range(2):
        data_params[i].time_params = {'relative': False, 't0': time_ranges[i][0], 'tf': time_ranges[i][1]}


    print(f"Loading pose data...")
    pose_data = []
    for i in range(2):
        os.environ[data_params[i].run_env] = args.runs[i]
        pose_data.append(data_params[i].load_pose_data())
        pose_data[-1].time_tol = 20.0

    print(f"Loading image data...")
    img_data = []
    for i in range(2):
        os.environ[data_params[i].run_env] = args.runs[i]
        img_data.append(data_params[i].load_img_data())

    fc = cv.VideoWriter_fourcc(*"mp4v")
    panel_width = img_data[0].camera_params.width
    panel_height = img_data[0].camera_params.height

    video = cv.VideoWriter(str(output_path), fc, vid_params.fps, (2*panel_width, 2*panel_height))

    # transform segments back into original odometry frame
    for i in range(2):
        T_odom_submap = submap_pair[i].pose_gravity_aligned
        for seg in submap_pair[i].segments:
            seg.transform(T_odom_submap)

    # get segment matches
    matched_segments = []
    # max_id = np.max([[seg.id for seg in submap_pair[i].segments] for i in range(2)])
    associated_objs = results.associated_objs_mat[idxs[0]][idxs[1]]
    for seg in submap_pair[0].segments + submap_pair[1].segments:
        seg.id += len(associated_objs)
    for i in range(len(associated_objs)):
        matched_segments.append((submap_pair[0].segments[associated_objs[i,0]], 
                                 submap_pair[1].segments[associated_objs[i,1]]))
        matched_segments[-1][0].id = i
        matched_segments[-1][1].id = i
        
    print("Creating video...")
    for t in np.arange(0.0, min_time_range, 1/vid_params.fps):
        
        viz_img = np.zeros((2*panel_height, 2*panel_width, 3), dtype=np.uint8)

        seg_seen = np.zeros((len(matched_segments), 2), dtype=bool)
        for i in range(2):
            t_i = time_ranges[i][0] + t
            img_i = img_data[i].img(t_i)
            
            for j in range(len(matched_segments)):
                seg = matched_segments[j][i]
                if np.linalg.norm(seg.center.flatten() - pose_data[i].position(t_i).flatten()) < vid_params.min_segment_dist:
                    img_i = visualize_segment_on_img(seg, pose_data[i].pose(t_i), img_i)
                    seg_seen[j, i] = True

            viz_img[panel_height*i:panel_height*(i+1), panel_width:] = img_i

        for j in range(len(matched_segments)):
            if np.all(seg_seen[j]):
                seg_pixels = []
                success = True
                for i in range(2):
                    new_outline = matched_segments[j][i].outline_2d(
                        pose_data[i].pose(time_ranges[i][0] + t))
                    if new_outline is None:
                        success = False
                        continue
                    seg_pixels.append(new_outline + np.array([panel_width, panel_height*i]))
                if not success:
                    continue
                nearest_pixels = (seg_pixels[0][0], seg_pixels[1][0])
                for pixels_ii in seg_pixels[0]:
                    for pixels_jj in seg_pixels[1]:
                        if np.linalg.norm(pixels_ii - pixels_jj) < \
                            np.linalg.norm(nearest_pixels[0] - nearest_pixels[1]):
                            nearest_pixels = (pixels_ii, pixels_jj)
                cv.line(viz_img, tuple(nearest_pixels[0].astype(np.int32)), 
                        tuple(nearest_pixels[1].astype(np.int32)), (0, 255, 0), 2)

        video.write(viz_img)

    video.release()
