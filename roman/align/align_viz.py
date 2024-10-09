import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List
import open3d as o3d

from robot_utils.robot_data.pose_data import PoseData
from robot_utils.transform import transform_2_xytheta
from robot_utils.geometry import circle_intersection

from segment_track.segment import Segment
from segment_track.tracker import Tracker

from object_map_registration.object.pointcloud_object import PointCloudObject

def create_ptcld_geometries(submap, color, submap_offset=np.array([0,0,0]), include_label=True):
    ocd_list = []
    label_list = []
    
    ptcldobj: PointCloudObject
    for ptcldobj in submap:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptcldobj.get_points())
        num_pts = ptcldobj.get_points().shape[0]
        rand_color = np.random.uniform(0, 1) * color
        rand_color = np.repeat(rand_color, num_pts, axis=0)
        pcd.colors = o3d.utility.Vector3dVector(rand_color)
        pcd.translate(submap_offset)
        ocd_list.append(pcd)
        
        if include_label:
            label = [f"id: {ptcldobj.id}", f"volume: {ptcldobj.volume:.2f}", 
                    f"extent: [{ptcldobj.extent[0]:.2f}, {ptcldobj.extent[1]:.2f}, {ptcldobj.extent[2]:.2f}]"]
            for i in range(3):
                label_list.append((np.median(pcd.points, axis=0) + 
                                np.array([0, 0, -0.15*i]), label[i]))
    
    return ocd_list, label_list

parser = argparse.ArgumentParser()
parser.add_argument('output_viz_file', type=str)
parser.add_argument('--idx', '-i', type=int, nargs=2, default=None)
parser.add_argument('--offset', type=float, nargs=3, default=[20.,0,0])
parser.add_argument('--no-text', action='store_true')
args = parser.parse_args()
output_viz_file = os.path.expanduser(args.output_viz_file)

pkl_file = open(output_viz_file, 'rb')
submaps_0, submaps_1, associated_objs_mat, robots_nearby_mat, T_ij_mat, T_ij_hat_mat = pickle.load(pkl_file)
pkl_file.close()
print(f'Loaded {len(submaps_0)} and {len(submaps_1)} submaps.')

if args.idx is None:
    clipper_num_associations  =  np.zeros((len(submaps_0), len(submaps_1)))*np.nan

    max_i = 0
    max_j = 0
    max_num = 0
    for i in range(len(submaps_0)):
        for j in range(len(submaps_1)):
            clipper_num_associations[i, j] =  len(associated_objs_mat[i][j])
            if len(associated_objs_mat[i][j]) > max_num:
                max_num = len(associated_objs_mat[i][j])
                max_i = i
                max_j = j

    plt.imshow(
        clipper_num_associations, 
        vmin=0, 
    )
    plt.colorbar(fraction=0.03, pad=0.04)
    plt.show()

# Same dir
# idx_0 = max_i+1
# idx_1 = max_j-1

# West Point
# idx_0 = 8
# # idx_1 = 27
# idx_1 = 28
# # idx_0 = max_i+1
# # idx_1 = max_j-1
# # print(idx_0)
# # print(idx_1)

# acl_jackal2/sparkal2 perpendicular
# idx

if args.idx is not None:
  idx_0, idx_1 = args.idx
else:
  idx_str = input("Please input two indices, separated by a space: \n")
  idx_0, idx_1 = [int(idx) for idx in idx_str.split()]


association = associated_objs_mat[idx_0][idx_1]
print(associated_objs_mat[idx_0][idx_1])
associated_objs_mat[idx_0-1][idx_1-1]
submap_0 = [PointCloudObject.from_pickle(data) for data in submaps_0[idx_0]]
submap_1 = [PointCloudObject.from_pickle(data) for data in submaps_1[idx_1]]
for obj in submap_0 + submap_1:
  obj.use_bottom_median_as_center()
print(f'Submap pair ({idx_0}, {idx_1}) contains {len(submap_0)} and {len(submap_1)} objects.')
print(f'Clipper finds {len(association)} associations.')

# Prepare submaps for visualization
edges = []
red_color = np.asarray([1,0,0]).reshape((1,3))
blue_color = np.asarray([0,0,1]).reshape((1,3))
seg0_color = red_color
seg1_color = blue_color
submap1_offset = np.asarray(args.offset)

ocd_list_0, label_list_0 = create_ptcld_geometries(submap_0, red_color, include_label=not args.no_text)
ocd_list_1, label_list_1 = create_ptcld_geometries(submap_1, blue_color, submap1_offset, include_label=not args.no_text)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

for obj_idx_0, obj_idx_1 in association:
    print(f'Add edge between {obj_idx_0} and {obj_idx_1}.')
    # points = [submap_0[obj_idx_0].center, submap_1[obj_idx_1].center]
    points = [ocd_list_0[obj_idx_0].get_center(), ocd_list_1[obj_idx_1].get_center()]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector([[0,1]]),
    )
    line_set.colors = o3d.utility.Vector3dVector([[0,1,0]])
    edges.append(line_set)

app = o3d.visualization.gui.Application.instance
app.initialize()
vis = o3d.visualization.O3DVisualizer()
vis.show_skybox(False)

for i, geom in enumerate(ocd_list_0 + ocd_list_1 + edges):
    vis.add_geometry(f"geom-{i}", geom)
for label in label_list_0 + label_list_1:
    vis.add_3d_label(*label)
vis.add_geometry("origin", origin)

vis.reset_camera_to_default()
app.add_window(vis)
app.run()