import copy
from typing import Dict

from evo.tools import plot
from evo.core import metrics
from evo.core import sync

from roman.offline_rpgo.g2o_and_time_to_pose_data import gt_csv_est_g2o_to_pose_data

def evaluate(est_g2o_file: str, est_time_file: str, gt_files: Dict[int, str]):
    pd_est, pd_gt = gt_csv_est_g2o_to_pose_data(
        est_g2o_file, est_time_file, gt_files)
        
    traj_ref = pd_gt.to_evo()
    traj_est = pd_est.to_evo()

    max_diff = 0.1

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)

    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

    pose_relation = metrics.PoseRelation.translation_part
    use_aligned_trajectories = True
    
    if use_aligned_trajectories:
        data = (traj_ref, traj_est_aligned) 
    else:
        data = (traj_ref, traj_est)
        
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    return ape_stat