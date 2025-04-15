import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List
import open3d as o3d

from robotdatapy.data import PoseData

from roman.object.pointcloud_object import PointCloudObject
from roman.align.results import SubmapAlignResults, plot_align_results
from roman.map.map import submaps_from_roman_map, ROMANMap, SubmapParams, Submap

def create_ptcld_geometries(submap: Submap, color=None, submap_offset=np.array([0,0,0]), 
                            include_label=True, transform=None, transform_gt=True):
    ocd_list = []
    label_list = []
    
    for seg in submap.segments:
        if transform is not None:
            seg.transform(transform)
        elif transform_gt and submap.has_gt:
            seg.transform(submap.pose_gravity_aligned_gt)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg.points)
        num_pts = seg.points.shape[0]
        if color is None:
            rand_color = np.array(seg.viz_color).reshape((1,3))/255.0
            seg_color = np.repeat(rand_color, num_pts, axis=0)
        else:
            seg_color = np.repeat(color, num_pts, axis=0)
        pcd.colors = o3d.utility.Vector3dVector(seg_color)
        pcd.translate(submap_offset)
        ocd_list.append(pcd)
        
        if include_label:
            label = [f"id: {seg.id}", f"volume: {seg.volume:.2f}"] 
                    # f"extent: [{ptcldobj.extent[0]:.2f}, {ptcldobj.extent[1]:.2f}, {ptcldobj.extent[2]:.2f}]"]
            for i in range(2):
                label_list.append((np.median(pcd.points, axis=0) + 
                                np.array([0, 0, -0.15*i]), label[i]))
    
    return ocd_list, label_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_viz_file', type=str)
    parser.add_argument('--idx', '-i', type=int, nargs=2, default=None)
    parser.add_argument('--offset', type=float, nargs=3, default=[0.,0.,10.])
    parser.add_argument('--text', action='store_true')
    parser.add_argument('--roman-maps', '-r', type=str, nargs=2, default=None)
    parser.add_argument('--gt', '-g', type=str, nargs=2, default=None)
    parser.add_argument('--align', '-a', action='store_true', default=False)
    parser.add_argument('--uniform-color', '-u', action='store_true', default=False)
    args = parser.parse_args()
    output_viz_file = os.path.expanduser(args.output_viz_file)

    # Load result data

    print('Loading data...')
    pkl_file = open(output_viz_file, 'rb')
    results: SubmapAlignResults
    results = pickle.load(pkl_file)

    gt_files = args.gt if args.gt is not None else results.submap_io.input_gt_pose_yaml
    roman_map_paths = args.roman_maps if args.roman_maps is not None else results.submap_io.inputs

    roman_maps = [ROMANMap.from_pickle(roman_map_paths[i]) for i in range(2)]
    submap_params = SubmapParams.from_submap_align_params(results.submap_align_params)
    submap_params.use_minimal_data = False
    gt_pose_data = []
    if gt_files != [None, None]:
        for i, yaml_file in enumerate(gt_files):
            if results.submap_io.robot_env is not None:
                os.environ[results.submap_io.robot_env] = results.submap_io.robot_names[i]
            if 'csv' in yaml_file:
                gt_pose_data.append(PoseData.from_kmd_gt_csv(yaml_file))
            else:
                gt_pose_data.append(PoseData.from_yaml(yaml_file))
            gt_pose_data[-1].time_tol = 100.0
    else:
        gt_pose_data = [None, None]

    submaps = [submaps_from_roman_map(
        roman_maps[i], submap_params, gt_pose_data[i]) for i in range(2)]
    pkl_file.close()
    print(f'Loaded {len(submaps[0])} and {len(submaps[1])} submaps.')

    # grab variables from results
    associated_objs_mat = results.associated_objs_mat
    T_ij_hat_mat = results.T_ij_hat_mat

    if args.idx is not None:
        idx_0, idx_1 = args.idx
    else:
        plot_align_results(results, dpi=100)
        plt.show()
        idx_str = input("Please input two indices, separated by a space: \n")
        idx_0, idx_1 = [int(idx) for idx in idx_str.split()]


    association = associated_objs_mat[idx_0][idx_1]
    associated_objs_mat[idx_0-1][idx_1-1]
    submap_0 = submaps[0][idx_0]
    submap_1 = submaps[1][idx_1]
    # for obj in submap_0 + submap_1:
    #   obj.use_bottom_median_as_center()
    print(f"Estimate distance error: {results.clipper_dist_mat[idx_0][idx_1]:.2f}")
    print(f"Estimate angle error: {results.clipper_angle_mat[idx_0][idx_1]:.2f}")
    print(f'Submap pair ({idx_0}, {idx_1}) contains {len(submap_0)} and {len(submap_1)} objects.')
    print(f'Clipper finds {len(association)} associations.')

    # Prepare submaps for visualization
    edges = []
    submap0_color = np.asarray([1,0,0]).reshape((1,3)) if args.uniform_color else None # red
    submap1_color = np.asarray([0,0,1]).reshape((1,3)) if args.uniform_color else None # blue
    if args.align:
        transform = T_ij_hat_mat[idx_0][idx_1]
        submap1_offset = np.zeros(3)
        transform_gt = False
    else:
        transform = None
        submap1_offset = np.asarray(args.offset)
        transform_gt = True

    ocd_list_0, label_list_0 = create_ptcld_geometries(submap_0, submap0_color, include_label=args.text, 
                                                       transform_gt=transform_gt)
    ocd_list_1, label_list_1 = create_ptcld_geometries(submap_1, submap1_color, submap1_offset, 
                                                       include_label=args.text, transform=transform, 
                                                       transform_gt=transform_gt)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    submap_origins = []
    for i, sm in enumerate([submap_0, submap_1]):
        submap_origins.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
        if sm.has_gt:
            submap_origins[-1].transform(sm.pose_gravity_aligned_gt)
    submap_origins[1].translate(submap1_offset)

    ids0, ids1 = [], []
    for obj_idx_0, obj_idx_1 in association:
        print(f'Add edge between {obj_idx_0} and {obj_idx_1}.')
        ids0.append(submap_0.segments[obj_idx_0].id)
        ids1.append(submap_1.segments[obj_idx_1].id)
        # points = [submap_0[obj_idx_0].center, submap_1[obj_idx_1].center]
        points = [ocd_list_0[obj_idx_0].get_center(), ocd_list_1[obj_idx_1].get_center()]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector([[0,1]]),
        )
        line_set.colors = o3d.utility.Vector3dVector([[0,1,0]])
        edges.append(line_set)
        
    print(f"Associated object ids, robot0: {ids0}")
    print(f"Associated object ids, robot1: {ids1}")

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer()
    vis.show_skybox(False)

    for i, geom in enumerate(ocd_list_0 + ocd_list_1 + edges + submap_origins):
        vis.add_geometry(f"geom-{i}", geom)
    for label in label_list_0 + label_list_1:
        vis.add_3d_label(*label)
    vis.add_geometry("origin", origin)

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()