import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pickle
import cv2 as cv
from dataclasses import dataclass
import os
from copy import deepcopy
import open3d as o3d
import tqdm

from roman.map.map import Submap, SubmapParams, submaps_from_roman_map, load_roman_map
from roman.align.results import SubmapAlignResults, plot_align_results, submaps_from_align_results
from roman.params.data_params import DataParams
from roman.viz import visualize_segment_on_img
from roman.align.align_viz import create_association_geometries, create_ptcld_geometries

from robotdatapy import transform

@dataclass
class VideoParams:

    fps: int = 10
    min_segment_dist: float = 15.0 # segment has to be within this distance to be drawn
    submap_z_diff: float = 10.0 # z offset to visualize two different submaps
    num_3d_spins: float = 1.0 # number of spins to do in 3D visualization
    dist_from_submap_center_3d: float = 25.0 # distance from submap center in 3D visualization
    img_ratio: float = 4/3 # width / height
    o3d_fov_deg: float = 60.0
    time_adjustments: tuple = (0.0, 0.0) # manual time adjustment for first and second submaps
    time_buffer: float = 1.0 # start and end this many seconds before and after the time range of the segments

def get_path(path_str) -> Path:
    path = Path(path_str).expanduser()
    return path

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--runs', '-r', type=str, nargs=2, required=True)
    parser.add_argument('--idx', '-i', type=int, nargs=2, default=None)
    parser.add_argument('--time-adjustments', '-t', type=float, nargs=2, default=[0.0, 0.0],
                        help='Manual time adjustment for the two submaps. Use this if the time ranges of the submap videos does not work well.')

    args = parser.parse_args()
    results_dir = get_path(args.results_dir)
    results_align_file = results_dir / 'align' / f'{args.runs[0]}_{args.runs[1]}' / 'align.pkl'
    params_dir = results_dir / 'params'
    output_path = get_path(args.output_path)
    vid_params = VideoParams()
    vid_params.time_adjustments = tuple(args.time_adjustments)


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
        
    # get segment matches
    submap_pair = [submaps[i][idxs[i]] for i in range(2)]
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

    # extract time range
    time_ranges = []
    for i in range(2):
        t0 = np.min([segs[i].first_seen for segs in matched_segments]) - vid_params.time_buffer
        tf = np.max([segs[i].last_seen for segs in matched_segments]) + vid_params.time_buffer
        time_ranges.append(np.array([t0, tf]))
    
    # make time ranges some duration
    min_time_range = np.min([time_ranges[i][1] - time_ranges[i][0] for i in range(2)])
    for i in range(2):
        time_range_i = time_ranges[i][1] - time_ranges[i][0]
        if time_range_i >= min_time_range:
            time_range_diff = time_range_i - min_time_range
            time_ranges[i][0] += time_range_diff/2
            time_ranges[i][1] -= time_range_diff/2
    
        for j in range(2):
            time_ranges[i][j] += vid_params.time_adjustments[i]

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
    img_w = img_data[0].camera_params.width
    img_h = img_data[0].camera_params.height
    o3d_h = img_data[0].camera_params.height * 2
    o3d_w = int(o3d_h * vid_params.img_ratio - img_w)

    video = cv.VideoWriter(str(output_path), fc, vid_params.fps, (o3d_w + img_w, o3d_h))
    
    submap_pair_in_submap_frame = deepcopy(submap_pair)

    # transform segments back into original odometry frame
    for i in range(2):
        T_odom_submap = submap_pair[i].pose_gravity_aligned
        for seg in submap_pair[i].segments:
            seg.transform(T_odom_submap)

    # transform other segment copy into same frame using Tij
    T_submapi_submapj = results.T_ij_hat_mat[idxs[0],idxs[1]]
    for seg in submap_pair_in_submap_frame[1].segments:
        seg.transform(T_submapi_submapj)
    # apply z offset to visualize two different submaps
    for seg in submap_pair_in_submap_frame[0].segments:
        seg.transform(transform.xyz_rpy_to_transform(np.array([0., 0., -vid_params.submap_z_diff]), 
                                                     np.array([0., 0., 0.])))
        
    # create point cloud geometries
    pcd_lists = [create_ptcld_geometries(submap_pair_in_submap_frame[i], color=None, 
        include_label=False, transform=None, transform_gt=False)[0] for i in range(2)]
    edges = create_association_geometries(pcd_lists[0], pcd_lists[1], associated_objs)

    renderer = o3d.visualization.rendering.OffscreenRenderer(o3d_w, o3d_h)
    scene = renderer.scene
    f = o3d_w / (2*np.tan(np.deg2rad(vid_params.o3d_fov_deg) / 2))
    o3d_K = np.array([[f, 0., o3d_w/2],
                          [0., f, o3d_h/2],
                          [0., 0., 1.]])

        
    print("Creating video...")
    for t in tqdm.tqdm(np.arange(0.0, min_time_range, 1/vid_params.fps)):
        
        viz_img = np.zeros((o3d_h, o3d_w + img_w, 3), dtype=np.uint8)

        seg_seen = np.zeros((len(matched_segments), 2), dtype=bool)
        # draw segments
        for i in range(2):
            t_i = time_ranges[i][0] + t
            img_i = img_data[i].img(t_i)
            
            for j in range(len(matched_segments)):
                seg = matched_segments[j][i]
                if np.linalg.norm(seg.center.flatten() - pose_data[i].position(t_i).flatten()) < vid_params.min_segment_dist:
                    img_i = visualize_segment_on_img(seg, pose_data[i].pose(t_i), img_i)
                    seg_seen[j, i] = True

            viz_img[img_h*i:img_h*(i+1), o3d_w:] = img_i

        # draw segment matches
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
                    seg_pixels.append(new_outline + np.array([o3d_w, img_h*i]))
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

        # show 3D visualization

        vid_percent = t / min_time_range
        camera_angle = vid_params.num_3d_spins * 360 * vid_percent
        camera_pose = transform.T_FLURDF @ transform.xyz_rpy_to_transform(
            np.array([0., 0., 0.]), np.array([0., camera_angle, 0.]), degrees=True)
        behind_camera = np.eye(4)
        behind_camera[:3, 3] = np.array([0., 5., -vid_params.dist_from_submap_center_3d])
        camera_pose = camera_pose @ behind_camera
        renderer.setup_camera(o3d_K, np.linalg.inv(camera_pose), o3d_w, o3d_h)

        edge_mat = o3d.visualization.rendering.MaterialRecord()
        edge_mat.shader = "unlitLine"
        edge_mat.line_width = 5.0
        pt_mat = o3d.visualization.rendering.MaterialRecord()
        pt_mat.point_size = 5.0

        for i, obj in enumerate(pcd_lists[0] + pcd_lists[1]):
            scene.add_geometry(f"pcd-{i}", obj, pt_mat)
        for i, edg in enumerate(edges):
            scene.add_geometry(f"edge-{i}", edg, edge_mat)
        o3d_img = renderer.render_to_image()
        o3d_img = cv.cvtColor(np.asarray(o3d_img), cv.COLOR_RGB2BGR)
        scene.clear_geometry()

        viz_img[:, :o3d_w] = o3d_img
        video.write(viz_img)

    video.release()
