# python3 ./demo/o3d_viz.py /home/masonbp/results/west_point_2023/segment_tracking/rgbd_2/sparkal2.pkl

import numpy as np
import open3d as o3d
import pickle
import argparse
import os

def main(args, filter_viz=False):

    with open(os.path.expanduser(args.pickle_file[0]), 'rb') as f:
        tracker, poses_history = pickle.load(f)
        
    poses_list = []
    pcd_list = []
    points_bounds = np.array([[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]])
    for seg in tracker.segments + tracker.segment_graveyard:
        if filter_viz:
            min_seg_id = 8900
            max_seg_id = 9600
            if not (seg.id > min_seg_id and seg.id < max_seg_id):
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
    for Twb in poses_history:
        if filter_viz and np.any(Twb[:3,3] < points_bounds[:,0]) or np.any(Twb[:3,3] > points_bounds[:,1]):
            continue
        pose_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        pose_obj.transform(Twb)
        poses_list.append(pose_obj)
    
    o3d.visualization.draw_geometries(poses_list + pcd_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_file', type=str, help='Path to pickle file', nargs=1)
    args = parser.parse_args()

    main(args, filter_viz=False)
