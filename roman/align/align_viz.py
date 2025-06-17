import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List, Tuple
import open3d as o3d
from dataclasses import dataclass

from robotdatapy.data import PoseData

from roman.object.segment import Segment
from roman.align.results import SubmapAlignResults, plot_align_results, submaps_from_align_results
from roman.map.map import submaps_from_roman_map, ROMANMap, SubmapParams, Submap

@dataclass
class AlignVizParams:
    offset: Tuple[float, float, float] = (0.,0.,10.)
    align: bool = False
    uniform_colors: Tuple[np.ndarray, np.ndarray] = \
        (np.asarray([1,0,0]).reshape((1,3)), np.asarray([0,0,1]).reshape((1,3)))
    use_uniform_colors: bool = False
    verbose: bool = False
    
@dataclass
class AlignVizGeometries:
    pointcloud_maps: Tuple[List[o3d.geometry.PointCloud], List[o3d.geometry.PointCloud]]
    edges: List[o3d.geometry.LineSet]
    submap_origins: List[o3d.geometry.TriangleMesh]
    labels: tuple
    
    @property
    def geometries(self):
        return self.pointcloud_maps[0] + self.pointcloud_maps[1] + self.edges + self.submap_origins
    
    @property
    def all_labels(self):
        return self.labels[0] + self.labels[1]

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

def create_association_geometries(pcd_list_0, pcd_list_1, associations: List[Tuple[int, int]],):
    # ids0, ids1 = [], []
    edges = []
    for obj_idx_0, obj_idx_1 in associations:
        # if params.verbose:
        #     print(f'Add edge between {obj_idx_0} and {obj_idx_1}.')
        # ids0.append(submap_0.segments[obj_idx_0].id)
        # ids1.append(submap_1.segments[obj_idx_1].id)
        # points = [submap_0[obj_idx_0].center, submap_1[obj_idx_1].center]
        points = [pcd_list_0[obj_idx_0].get_center(), pcd_list_1[obj_idx_1].get_center()]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector([[0,1]]),
        )
        line_set.colors = o3d.utility.Vector3dVector([[0,1,0]])
        edges.append(line_set)
    
    # if params.verbose:
    #     print(f"Associated object ids, robot0: {ids0}")
    #     print(f"Associated object ids, robot1: {ids1}")
    return edges

def align_viz(
    submaps: Tuple[List[Submap], List[Submap]], 
    idxs: Tuple[int, int], 
    results: SubmapAlignResults, 
    params: AlignVizParams = AlignVizParams()
) -> AlignVizGeometries:
    
    # Prepare submaps for visualization
    association = results.associated_objs_mat[idxs[0]][idxs[1]]
    results.associated_objs_mat[idxs[0]-1][idxs[1]-1]
    submap_0 = submaps[0][idxs[0]]
    submap_1 = submaps[1][idxs[1]]
    # for obj in submap_0 + submap_1:
    #   obj.use_bottom_median_as_center()
    if params.verbose:
        print(f"Estimate distance error: {results.clipper_dist_mat[idxs[0]][idxs[1]]:.2f}")
        print(f"Estimate angle error: {results.clipper_angle_mat[idxs[0]][idxs[1]]:.2f}")
        print(f'Submap pair ({idxs[0]}, {idxs[1]}) contains {len(submap_0)} and {len(submap_1)} objects.')
        print(f'Clipper finds {len(association)} associations.')

    # Prepare submaps for visualization
    submap0_color = params.uniform_colors[0] if params.use_uniform_colors else None # red
    submap1_color = params.uniform_colors[1] if params.use_uniform_colors else None # blue
    if params.align:
        transform = results.T_ij_hat_mat[idxs[0]][idxs[1]]
        submap1_offset = np.zeros(3)
        transform_gt = False
    else:
        transform = None
        submap1_offset = np.asarray(params.offset)
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
        if not args.align and sm.has_gt:
            submap_origins[-1].transform(sm.pose_gravity_aligned_gt)
    if not args.align:
        submap_origins[1].translate(submap1_offset)
    else:
        submap_origins[1].transform(transform)

    edges = create_association_geometries(ocd_list_0, ocd_list_1, association)
        
    return AlignVizGeometries(
        pointcloud_maps=(ocd_list_0, ocd_list_1),
        edges=edges,
        submap_origins=submap_origins,
        labels=(label_list_0, label_list_1),
    )

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
    pkl_file.close()
    submaps = submaps_from_align_results(results, args.gt, args.roman_maps)

    print(f'Loaded {len(submaps[0])} and {len(submaps[1])} submaps.')

    if args.idx is not None:
        idxs = args.idx
    else:
        plot_align_results(results, dpi=100)
        plt.show()
        idx_str = input("Please input two indices, separated by a space: \n")
        idxs = [int(idx) for idx in idx_str.split()]

    params = AlignVizParams(
        offset=args.offset,
        use_uniform_colors=args.uniform_color,
    )
    geometries = align_viz(submaps, idxs, results, params)

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer()
    vis.show_skybox(False)

    for i, geom in enumerate(geometries.geometries):
        vis.add_geometry(f"geom-{i}", geom)
    for label in geometries.all_labels:
        vis.add_3d_label(*label)
    # vis.add_geometry("origin", origin)

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()
