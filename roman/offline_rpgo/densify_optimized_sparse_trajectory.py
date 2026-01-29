import argparse
import numpy as np
import robotdatapy as rdp
import gtsam
from scipy.spatial.transform import Rotation as Rot
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import tqdm
import os
from roman.utils import expandvars_recursive

from roman.params.data_params import DataParams

def T_4x4_to_gtsam(T):
    """Convert a 4x4 homogeneous transformation matrix to a gtsam Pose3."""
    T_q_wxyz = [Rot.from_matrix(T[:3, :3]).as_quat()[i] for i in [3, 0, 1, 2]]
    T_t = T[:3, 3]
    return gtsam.Pose3(gtsam.Rot3.Quaternion(*T_q_wxyz), gtsam.Point3(T_t))

def gtsam_noise_from_sigmas(rpy_sigma, xyz_sigma):
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.array([rpy_sigma, rpy_sigma, rpy_sigma, xyz_sigma, xyz_sigma, xyz_sigma]))

def densify_optimized_sparse_trajectory(
        dense_unoptimized: rdp.data.PoseData,
        sparse_optimized: rdp.data.PoseData,
        dense_rpy_sigma: float = np.deg2rad(0.1),
        dense_xyz_sigma: float = 0.02,
        sparse_rpy_sigma: float = np.deg2rad(5.0),
        sparse_xyz_sigma: float = 0.5,
    ) -> rdp.data.PoseData:
    """
    Convert a sparse optimized trajectory into a dense optimized trajectory using the 
    sparse trajectory as priors in a pose graph with the dense unoptimized trajectory as odometry.

    Args:
        dense_unoptimized (rdp.data.PoseData): Dense unoptimized trajectory.
        sparse_optimized (rdp.data.PoseData): Sparse optimized trajectory.  
        dense_rpy_sigma (float): Roll-pitch-yaw standard deviation for dense factors.
        dense_xyz_sigma (float): X-Y-Z standard deviation for dense factors.
        sparse_rpy_sigma (float): Roll-pitch-yaw standard deviation for sparse factors.
        sparse_xyz_sigma (float): X-Y-Z standard deviation for sparse factors.

    Returns:
        rdp.data.PoseData: Dense optimized trajectory.
    """
    sparse_noise = gtsam_noise_from_sigmas(sparse_rpy_sigma, sparse_xyz_sigma)
    dense_noise = gtsam_noise_from_sigmas(dense_rpy_sigma, dense_xyz_sigma)
    graph = gtsam.NonlinearFactorGraph()

    T_d_iminus1 = dense_unoptimized.pose(dense_unoptimized.times[0])
    graph.add(gtsam.PriorFactorPose3(0, T_4x4_to_gtsam(sparse_optimized.pose(dense_unoptimized.times[0])), sparse_noise))

    # sparse_optimized.time_tol = 0.25
    for i, t_i in enumerate(dense_unoptimized.times):
        if i == 0:
            continue
        T_d_i = dense_unoptimized.pose(dense_unoptimized.times[i])
        T_iminus1_i = np.linalg.inv(T_d_iminus1) @ T_d_i
        T_iminus1_i_gtsam = T_4x4_to_gtsam(T_iminus1_i)
        graph.add(gtsam.BetweenFactorPose3(i-1, i, T_iminus1_i_gtsam, dense_noise))
        T_d_iminus1 = T_d_i

    for i, t_i in enumerate(dense_unoptimized.times):  
        try:
            T_s_i = sparse_optimized.pose(t_i)
            graph.add(gtsam.PriorFactorPose3(i, T_4x4_to_gtsam(T_s_i), sparse_noise))
        except rdp.exceptions.NoDataNearTimeException as e:
            continue
        

    initial_estimate = gtsam.Values()
    for i, t_i in enumerate(dense_unoptimized.times):
        T_w_i_gtsam = T_4x4_to_gtsam(sparse_optimized.pose(t_i))
        initial_estimate.insert(i, T_w_i_gtsam)
        
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    result_pd = rdp.data.PoseData.from_times_and_poses(
        dense_unoptimized.times, [result.atPose3(i).matrix() for i in range(len(dense_unoptimized.times))], time_tol=10.0)
    return result_pd

def densify_offline_rpgo_output(
    data_params_file: str,
    roman_output_dir: str,
    child_frame: str = 'flu',
    output_path_format: str = None,
    run: str = None,
    **kwargs
):
    assert child_frame in ['flu', 'camera'], "child_frame must be 'flu' or 'camera'"

    data_params = DataParams.from_yaml(data_params_file)
    runs = [run] if run is not None else data_params.runs
    for _, run in enumerate(runs):
        if data_params.run_env is not None:
            os.environ[data_params.run_env] = run

        data_params_i = DataParams.from_yaml(data_params_file, run=run)

        odometry_data = data_params_i.load_pose_data()

        if data_params_i.pose_data_params.T_camera_flu is not None:
            odometry_data.T_postmultiply = \
                odometry_data.T_postmultiply @ data_params_i.pose_data_params.T_camera_flu

        rpgo_output_trajectory_path = expandvars_recursive(
            f"{roman_output_dir}/offline_rpgo/{run}.csv")
        
        rpgo_output_trajectory = rdp.data.PoseData.from_kmd_gt_csv(
            rpgo_output_trajectory_path, time_tol=100.0)
        
        densified_data_flu = densify_optimized_sparse_trajectory(
            dense_unoptimized=odometry_data,
            sparse_optimized=rpgo_output_trajectory,
            **kwargs
        )

        if child_frame == 'camera' and data_params_i.pose_data_params.T_camera_flu is not None:
            densified_data_flu.T_postmultiply = \
                densified_data_flu.T_postmultiply @ np.linalg.inv(data_params_i.pose_data_params.T_camera_flu)
            
        if output_path_format is None:
            output_path_format = "{}" + f"_densified_{child_frame}.csv"
        output_path = expandvars_recursive(
            f"{roman_output_dir}/offline_rpgo/" + output_path_format.format(run))
        
        densified_data_flu.to_csv(output_path)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, required=True,
                        help='Path to ROMAN params directory.')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Path to ROMAN output directory.')
    parser.add_argument('--child-frame', type=str, default='flu',
                        help="Child frame of output trajectory, either 'flu' or 'camera'.")
    
    args = parser.parse_args()
    
    data_params_file = expandvars_recursive(os.path.join(args.params, 'data.yaml'))
    densify_offline_rpgo_output(
        data_params_file=data_params_file,
        roman_output_dir=args.output_dir,
        child_frame=args.child_frame
    )