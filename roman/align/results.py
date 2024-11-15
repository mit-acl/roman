import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass
from typing import List
import json

from robotdatapy.transform import transform_to_xytheta, transform_to_xyz_quat, \
    transform_to_xyzrpy

from roman.utils import transform_rm_roll_pitch
from roman.align.params import SubmapAlignInputOutput, SubmapAlignParams

from roman.object.segment import Segment

@dataclass
class SubmapAlignResults:
    robots_nearby_mat: np.array
    clipper_angle_mat: np.array
    clipper_dist_mat: np.array
    clipper_num_associations: np.array
    submap_yaw_diff_mat: np.array
    associated_objs_mat: np.array
    T_ij_mat: np.array
    T_ij_hat_mat: np.array
    timing_list: List[float]

def time_to_secs_nsecs(t, as_dict=False):
    seconds = int(t)
    nanoseconds = int((t - int(t)) * 1e9)
    if not as_dict:
        return seconds, nanoseconds
    else:
        return {'seconds': seconds, 'nanoseconds': nanoseconds}

def save_submap_align_results(sm_params: SubmapAlignParams, sm_io: SubmapAlignInputOutput, submaps, results: SubmapAlignResults,
                              trackers, times, submap_idxs, submap_centers):
    # Create plots
    fig, ax = plt.subplots(1, 5, figsize=(20, 5), dpi=500)
    fig.subplots_adjust(wspace=.3)
    fig.suptitle(sm_io.run_name)

    mp = ax[0].imshow(results.robots_nearby_mat, cmap='viridis', vmin=0)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[0].set_title("Submaps Overlap")

    mp = ax[1].imshow(-results.clipper_angle_mat, cmap='viridis', vmax=0, vmin=-10)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[1].set_title("Registration Error (deg)")

    mp = ax[2].imshow(-results.clipper_dist_mat, cmap='viridis', vmax=0, vmin=-5.0)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[2].set_title("Registration Distance Error (m)")

    mp = ax[3].imshow(results.clipper_num_associations, cmap='viridis', vmin=0)
    fig.colorbar(mp, fraction=0.03, pad=0.04)
    ax[3].set_title("Number of CLIPPER Associations")

    mp = ax[4].imshow(results.submap_yaw_diff_mat, cmap='viridis', vmin=0)
    fig.colorbar(mp, fraction=0.04, pad=0.04)
    ax[4].set_title("Submap Yaw Difference (deg)")

    for i in range(len(ax)):
        ax[i].set_xlabel("submap index (robot 2)")
        ax[i].set_ylabel("submap index (robot 1)")
        ax[i].grid(False)

    plt.savefig(sm_io.output_img)
        
    # for saving matrix results instead of image
    pkl_file = open(sm_io.output_matrix, 'wb')
    pickle.dump([results.robots_nearby_mat, results.clipper_angle_mat, results.clipper_dist_mat, 
                 results.clipper_num_associations, results.submap_yaw_diff_mat], pkl_file)
    pkl_file.close()
        
    # stores the submaps, associated objects, ground truth object overlap, and ground truth and estimated submap transformations
    submaps_pkl = [[[obj.to_pickle() for obj in sm] for sm in sms] for sms in submaps]
    pkl_file = open(sm_io.output_viz, 'wb')
    # submaps for robot i, submaps for robot j, associated object indices, matrix of how much overlap is between submaps, ground truth T_ij, estimated T_ij
    pickle.dump([submaps_pkl[0], submaps_pkl[1], results.associated_objs_mat, results.robots_nearby_mat, results.T_ij_mat, results.T_ij_hat_mat], pkl_file)
    pkl_file.close()

    with open(sm_io.output_timing, 'w') as f:
        f.write(f"Total number of submaps: {len(submaps[0])} x {len(submaps[1])} = {len(submaps[0])*len(submaps[1])}\n")
        f.write(f"Average time per registration: {np.mean(results.timing_list):.4f} seconds\n")
        f.write(f"Total time: {np.sum(results.timing_list):.4f} seconds\n")
        f.write(f"Total number of objects: {np.sum([len(submap) for submap in submaps[0] + submaps[1]])}\n")
        f.write(f"Average number of obects per map: {np.mean([len(submap) for submap in submaps[0] + submaps[1]]):.2f}\n")
    
    with open(sm_io.output_params, 'w') as f:
        f.write(f"{sm_params}")

    I_t = 1 / (sm_io.g2o_t_std**2)
    I_r = 1 / (sm_io.g2o_r_std**2)
    I = np.diag([I_t, I_t, I_t, I_r, I_r, I_r])
    
    json_output = []

    with open(sm_io.output_g2o, 'w') as f:
        for i in range(len(submaps[0])):
            for j in range(len(submaps[1])):
                if results.clipper_num_associations[i, j] < sm_io.lc_association_thresh:
                    continue
                if (np.abs(times[0][submap_idxs[0][i]] - times[1][submap_idxs[1][j]]) < 
                    sm_params.single_robot_lc_time_thresh and sm_params.single_robot_lc):
                    continue
                T_ci_cj = results.T_ij_hat_mat[i, j] # transform from center_j to center_i
                T_odomi_ci = transform_rm_roll_pitch(submap_centers[0][i])
                T_odomj_cj = transform_rm_roll_pitch(submap_centers[1][j])
                T_odomi_pi = submap_centers[0][i] # pose i in odom frame
                T_odomj_pj = submap_centers[1][j] # pose j in odom frame
                T_pi_pj = ( # pose j in pose i frame, the desired format for our loop closure
                    np.linalg.inv(T_odomi_pi) @ T_odomi_ci @ T_ci_cj @ np.linalg.inv(T_odomj_cj) @ T_odomj_pj
                )
                t, q = transform_to_xyz_quat(T_pi_pj, separate=True)
                json_output.append({
                    'seconds': [int(times[0][submap_idxs[0][i]]), int(times[1][submap_idxs[1][j]])],
                    'nanoseconds': [int((times[0][submap_idxs[0][i]] % 1) * 1e9), int((times[1][submap_idxs[1][j]] % 1) * 1e9)],
                    'names': sm_io.robot_names,
                    'translation': t.tolist(),
                    'rotation': q.tolist(),
                    'rotation_convention': 'xyzw',
                })

                f.write(f"# LC: {int(results.clipper_num_associations[i, j])}\n")
                f.write(f"EDGE_SE3:QUAT a{submap_idxs[0][i]} b{submap_idxs[1][j]} \t")
                f.write(f"{t[0]} {t[1]} {t[2]} \t")
                f.write(f"{q[0]} {q[1]} {q[2]} {q[3]} \t")
                for ii in range(6):
                    for jj in range(6):
                        if jj < ii:
                            continue
                        f.write(f"{I[ii, jj]} ")
                    f.write("\t")
                f.write("\n")
        f.close()
            
        with open(sm_io.output_lc_json, 'w') as f:
            json.dump(json_output, f, indent=4)
            f.close()
            
        for i, output_sm in enumerate(sm_io.output_submaps):
            tracker = trackers[i]
            if output_sm is not None:
                with open(output_sm, 'w') as f:
                    sm_json = dict()
                    sm_json['segments'] = []
                    sm_json['submaps'] = []
                    
                    segment: Segment
                    for segment in tracker.segment_graveyard + tracker.inactive_segments + tracker.segments:
                        try:
                            segment_json = {}
                            segment_json['robot_name'] = sm_io.robot_names[i]
                            segment_json['segment_index'] = segment.id
                            segment_json['centroid_odom'] = np.mean(segment.points, axis=0).tolist()
                            e = segment.normalized_eigenvalues()
                            segment_json['shape_attributes'] = {'volume': segment.volume, 
                                                                'linearity': segment.linearity(e), 
                                                                'planarity': segment.planarity(e), 
                                                                'scattering': segment.scattering(e)}
                            segment_json['first_seen'] = time_to_secs_nsecs(segment.first_seen, as_dict=True)
                            segment_json['last_seen'] = time_to_secs_nsecs(segment.last_seen, as_dict=True)
                            sm_json['segments'].append(segment_json)
                        except:
                            continue
                        
                    for j in range(len(submaps[i])):
                        t_j = times[i][submap_idxs[i][j]]
                        xyzquat_submap = transform_to_xyz_quat(submap_centers[i][j], separate=False)
                        sm_json['submaps'].append({
                            'submap_index': j,
                            'T_odom_submap': {
                                'tx': xyzquat_submap[0],
                                'ty': xyzquat_submap[1],
                                'tz': xyzquat_submap[2],
                                'qx': xyzquat_submap[3],
                                'qy': xyzquat_submap[4],
                                'qz': xyzquat_submap[5],
                                'qw': xyzquat_submap[6],
                            },
                            'robot_name': sm_io.robot_names[i],
                            'seconds': int(t_j),
                            'nanoseconds': int((t_j % 1) * 1e9),
                            'segment_indices': [obj.id for obj in submaps[i][j]]
                        })
                    json.dump(sm_json, f, indent=4)
                    f.close()