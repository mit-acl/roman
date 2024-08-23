# python3 ./demo/o3d_viz.py /home/masonbp/results/west_point_2023/segment_tracking/rgbd_2/sparkal2.pkl
# helpful for figuring out open3d visualization: 
# https://github.com/isl-org/Open3D/pull/3233/files#diff-4bf889278ab3bd4bb37dafdc436b6b6bc772b5e730dcd67a5e7e3f369cee694c

import numpy as np
import open3d as o3d
import pickle
import argparse
import os

def main(args, filter_viz=False):

    with open(os.path.expanduser(args.pickle_file[0]), 'rb') as f:
        pickle_data = pickle.load(f)
        if len(pickle_data) == 2:
            tracker, poses_history, = pickle_data
            times = None
        else:
            tracker, poses_history, times = pickle_data
        
    poses_list = []
    pcd_list = []
    label_list = []
    points_bounds = np.array([[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]])
    if args.time_range is not None:
        args.time_range = np.array(args.time_range) + times[0]

    for seg in tracker.segments + tracker.inactive_segments + tracker.segment_graveyard:
        if filter_viz:
            min_seg_id = 8900
            max_seg_id = 9600
            if not (seg.id > min_seg_id and seg.id < max_seg_id):
                continue
        if args.time_range is not None:
            if seg.first_seen > args.time_range[1] or seg.last_seen < args.time_range[0]:
                continue
        seg_points = seg.points
        if seg_points is not None:
            for i in range(3):
                points_bounds[i, 0] = min(points_bounds[i, 0], np.min(seg_points[:, i]))
                points_bounds[i, 1] = max(points_bounds[i, 1], np.max(seg_points[:, i]))
            num_pts = seg_points.shape[0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(seg_points)
            rand_color = np.random.uniform(0, 1, size=(1,3))
            rand_color = np.repeat(rand_color, num_pts, axis=0)
            pcd.colors = o3d.utility.Vector3dVector(rand_color)
            pcd_list.append(pcd)
            if not args.no_text:
                # label = [f"id: {seg.id}", f"volume: {seg.volume():.2f}", 
                #         f"extent: [{seg.extent[0]:.2f}, {seg.extent[1]:.2f}, {seg.extent[2]:.2f}]"]
                label = [f"id: {seg.id}"]
                for i in range(len(label)):
                    label_list.append((np.median(pcd.points, axis=0) + np.array([0, 0, -0.15*i]), label[i]))
                    
    print(f"Displaying {len(pcd_list)} objects.")

    displayed_positions = []
    for i, Twb in enumerate(poses_history):
        if filter_viz and (np.any(Twb[:3,3] < points_bounds[:,0]) or np.any(Twb[:3,3] > points_bounds[:,1])):
            continue
        if args.time_range is not None:
            if times is not None:
                t = times[i]
                if t < args.time_range[0] or t > args.time_range[1]:
                    continue
        
        displayed_positions.append(Twb[:3,3])
        pose_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        pose_obj.transform(Twb)
        poses_list.append(pose_obj)

    if not args.no_orig:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        poses_list.append(origin)
    else:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        T = np.eye(4)
        T[:3,3] = np.mean(displayed_positions, axis=0)
        origin.transform(T)
        poses_list.append(origin)
    
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer()
    # vis.show_skybox(False)

    mat = o3d.visualization.rendering.Material()
    mat.shader = 'defaultUnlit'
    mat.point_size = 5.0

    for i, obj in enumerate(pcd_list):
        vis.add_geometry(f"pcd-{i}", obj, mat)
    for label in label_list:
        vis.add_3d_label(*label)
    # for i, obj in enumerate(poses_list):
    #     vis.add_geometry(f"pose-{i}", obj)

    K = np.array([[200, 0, 200],
                [0, 200, 200],
                [0, 0, 1]]).astype(np.float64)
    if not args.no_orig:
        T_inv = np.array([[1,   0,  0, 0],
                        [0,   0,  1, -5],
                        [0,   -1, 0, 0],
                        [0,   0,  0, 1]]).astype(np.float64)
    else:
        mean_position = np.mean(displayed_positions, axis=0)
        T_inv = np.array([[1,   0,  0, 0],
                        [0,   -1,  0, 0],
                        [0,   0, -1, 20],
                        [0,   0,  0, 1]]).astype(np.float64)
        T_inv[:3,3] += mean_position
    T = np.linalg.inv(T_inv)
    vis.setup_camera(K, T, 400, 400)
    app.add_window(vis)
    app.run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_file', type=str, help='Path to pickle file', nargs=1)
    parser.add_argument('--no-text', action='store_true', help='Do not display text labels')
    parser.add_argument('--no-orig', action='store_true', help='Do not display origin')
    parser.add_argument('-t', '--time-range', type=float, nargs=2, help='Time range to display')
    args = parser.parse_args()

    main(args, filter_viz=False)
