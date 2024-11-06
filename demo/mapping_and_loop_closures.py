import numpy as np
import matplotlib.pyplot as plt

import argparse
from typing import List
import os

from roman.align.params import SubmapAlignInputOutput, SubmapAlignParams
from roman.align.submap_align import submap_align
from roman.offline_rpgo.extract_odom_g2o import roman_map_pkl_to_g2o
from roman.offline_rpgo.g2o_file_fusion import create_config, g2o_file_fusion
from roman.offline_rpgo.combine_loop_closures import combine_loop_closures
from roman.offline_rpgo.plot_g2o import plot_g2o, DEFAULT_TRAJECTORY_COLORS, G2OPlotParams
from roman.offline_rpgo.g2o_and_time_to_pose_data import g2o_and_time_to_pose_data
from roman.offline_rpgo.evaluate import evaluate
from roman.offline_rpgo.edit_g2o_edge_information import edit_g2o_edge_information

import demo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runs', type=str, nargs="+", 
                        help='Run names. Individual map files will be saved using these names.')
    parser.add_argument('-p', '--params', default=None, type=str, help='Path to params file', required=False)
    parser.add_argument('--params-list', type=str, default=None, nargs='+', help='Path to multiple params files')
    parser.add_argument('-o', '--output-dir', type=str, help='Path to output directory', required=True, default=None)
    parser.add_argument('-m', '--viz-map', action='store_true', help='Visualize map')
    parser.add_argument('-v', '--viz-observations', action='store_true', help='Visualize observations')
    parser.add_argument('--vid-rate', type=float, help='Video playback rate', default=1.0)
    parser.add_argument('-d', '--save-img-data', action='store_true', help='Save video frames as ImgData class')
    parser.add_argument('-s', '--sparse-pgo', action='store_true', help='Use sparse pose graph optimization')
    parser.add_argument('-n', '--num-req-assoc', type=int, help='Number of required associations', default=4)
    # parser.add_argument('--set-env-vars', type=str)
    parser.add_argument('--gt', type=str, nargs='+', help='Paths to ground truth pose yaml file')

    parser.add_argument('--skip-map', action='store_true', help='Skip mapping')
    parser.add_argument('--skip-align', action='store_true', help='Skip alignment')
    parser.add_argument('--skip-rpgo', action='store_true', help='Skip robust pose graph optimization')
    parser.add_argument('--skip-indices', type=int, nargs='+', help='Skip specific runs in mapping and alignment')
    
    args = parser.parse_args()

    # create output directories
    os.makedirs(os.path.join(args.output_dir, "map"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "align"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "offline_rpgo"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "offline_rpgo/sparse"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "offline_rpgo/dense"), exist_ok=True)
    
    assert args.sparse_pgo, "Only sparse pose graph optimization currently supported."
        
    if not args.skip_map:
        args.viz_open3d = False
        
        assert args.params is not None or args.params_list is not None, \
            "Either --params or --params-list must be provided."
        if args.params_list:
            assert len(args.params_list) == len(args.runs), \
                "Number of params files must match number of runs."
        
        for i, run in enumerate(args.runs):
            if args.skip_indices and i in args.skip_indices:
                continue
            if args.params_list:
                args.params = args.params_list[i]
            # mkdir $output_dir/map
            args.output = os.path.join(args.output_dir, "map", f"{run}")
                
            # shell: export RUN=run
            os.environ['RUN'] = run
            
            print(f"Mapping: {run}")
            demo.main(args)
        
    if not args.skip_align:
        sm_params = SubmapAlignParams()
        # TODO: support ground truth pose file for validation
        # if args.gt:
        #     sm_io.input_gt_pose_yaml = args.gt
            
        for i in range(len(args.runs)):
            if args.skip_indices and i in args.skip_indices:
                continue
            for j in range(i, len(args.runs)):
                if args.skip_indices and j in args.skip_indices:
                    continue
                output_dir = os.path.join(args.output_dir, "align", f"{args.runs[i]}_{args.runs[j]}")
                os.makedirs(output_dir, exist_ok=True)
                input_files = [os.path.join(args.output_dir, "map", f"{args.runs[i]}.pkl"),
                            os.path.join(args.output_dir, "map", f"{args.runs[j]}.pkl")]
                sm_io = SubmapAlignInputOutput(
                    inputs=input_files,
                    output_dir=output_dir,
                    run_name="align",
                    lc_association_thresh=args.num_req_assoc,
                    input_gt_pose_yaml=[args.gt[i], args.gt[j]] if args.gt is not None else [None, None],
                    robot_names=[args.runs[i], args.runs[j]],
                )
                sm_params.single_robot_lc = (i == j)
                submap_align(sm_params=sm_params, sm_io=sm_io)
                
    t_std = 0.005 if not args.sparse_pgo else 0.1
    r_std = np.deg2rad(0.025) if not args.sparse_pgo else np.deg2rad(0.5)
    min_keyframe_dist = 0.01 if not args.sparse_pgo else 2.0
                
    if not args.skip_rpgo:
        # Create g2o files for odometry
        for i, run in enumerate(args.runs):
            roman_map_pkl_to_g2o(
                pkl_file=os.path.join(args.output_dir, "map", f"{run}.pkl"),
                g2o_file=os.path.join(args.output_dir, "offline_rpgo/sparse", f"{run}.g2o"),
                time_file=os.path.join(args.output_dir, "offline_rpgo/sparse", f"{run}.time.txt"),
                robot_id=i,
                min_keyframe_dist=min_keyframe_dist,
                t_std=t_std,
                r_std=r_std,
                verbose=True
            )
            
            # create dense g2o file
            roman_map_pkl_to_g2o(
                pkl_file=os.path.join(args.output_dir, "map", f"{run}.pkl"),
                g2o_file=os.path.join(args.output_dir, "offline_rpgo/dense", f"{run}.g2o"),
                time_file=os.path.join(args.output_dir, "offline_rpgo/dense", f"{run}.time.txt"),
                robot_id=i,
                min_keyframe_dist=None,
                t_std=0.005,
                r_std=np.deg2rad(0.025),
                verbose=True
            )
        
        # Combine timing files
        odom_sparse_all_time_file = os.path.join(args.output_dir, "offline_rpgo/sparse", "odom_all.time.txt")
        odom_dense_all_time_file = os.path.join(args.output_dir, "offline_rpgo/dense", "odom_all.time.txt")
        with open(odom_sparse_all_time_file, 'w') as f:
            for i, run in enumerate(args.runs):
                with open(os.path.join(args.output_dir, "offline_rpgo/sparse", f"{run}.time.txt"), 'r') as f2:
                    f.write(f2.read())
                f2.close()
        with open(odom_dense_all_time_file, 'w') as f:
            for i, run in enumerate(args.runs):
                with open(os.path.join(args.output_dir, "offline_rpgo/dense", f"{run}.time.txt"), 'r') as f2:
                    f.write(f2.read())
                f2.close()
        
        # Fuse all odometry g2o files
        odom_sparse_all_g2o_file = os.path.join(args.output_dir, "offline_rpgo/sparse", "odom_all.g2o")
        g2o_fusion_config = create_config(robots=args.runs, 
            odometry_g2o_dir=os.path.join(args.output_dir, "offline_rpgo/sparse"))
        g2o_file_fusion(g2o_fusion_config, odom_sparse_all_g2o_file, thresh=args.num_req_assoc)
        
        # Fuse dense g2o file including loop closures
        dense_g2o_file = os.path.join(args.output_dir, "offline_rpgo/dense", "odom_and_lc.g2o")
        g2o_fusion_config = create_config(robots=args.runs, 
            odometry_g2o_dir=os.path.join(args.output_dir, "offline_rpgo/dense"),
            submap_align_dir=os.path.join(args.output_dir, "align"), align_file_name="align")
        g2o_file_fusion(g2o_fusion_config, dense_g2o_file, thresh=args.num_req_assoc)

        # Add loop closures to odometry g2o files
        final_g2o_file = os.path.join(args.output_dir, "offline_rpgo", "odom_and_lc.g2o")
        combine_loop_closures(
            g2o_reference=odom_sparse_all_g2o_file, 
            g2o_extra_lc=dense_g2o_file, 
            vertex_times_reference=odom_sparse_all_time_file,
            vertex_times_extra_lc=odom_dense_all_time_file,
            output_file=final_g2o_file,
        )
        
        # change lc covar
        # with open(os.path.expanduser(final_g2o_file), 'r') as f:
        #     g2o_lines = f.readlines()
        # # g2o_lines = edit_g2o_edge_information(g2o_lines, 0.1, np.deg2rad(0.5), loop_closures=True)
        # g2o_lines = edit_g2o_edge_information(g2o_lines, 0.25, np.deg2rad(0.5), loop_closures=True)
        # with open(os.path.expanduser(final_g2o_file), 'w') as f:
        #     for line in g2o_lines:
        #         f.write(line + '\n')
        #     f.close()
            
        # run kimera centralized robust pose graph optimization
        result_g2o_file = os.path.join(args.output_dir, "offline_rpgo", "result.g2o")
        ros_launch_command = f"roslaunch kimera_centralized_pgmo offline_g2o_solver.launch \
            g2o_file:={final_g2o_file} \
            output_path:={os.path.join(args.output_dir, 'offline_rpgo')}"
        os.system(ros_launch_command)
        
        # plot results
        g2o_symbol_to_name = {chr(97 + i): args.runs[i] for i in range(len(args.runs))}
        g2o_plot_params = G2OPlotParams()
        fig, ax = plt.subplots(3, 1)
        for i in range(3):
            g2o_plot_params.axes = [(0, 1), (0, 2), (1, 2)][i]
            plot_g2o(
                g2o_path=result_g2o_file,
                g2o_symbol_to_name=g2o_symbol_to_name,
                g2o_symbol_to_color=DEFAULT_TRAJECTORY_COLORS,
                ax=ax[i],
                params=g2o_plot_params
            )
        plt.savefig(os.path.join(args.output_dir, "offline_rpgo", "result.png"))
        print(f"Results saved to {os.path.join(args.output_dir, 'offline_rpgo', 'result.png')}")
        
        # Save csv files with resulting trajectories
        for i, run in enumerate(args.runs):
            pose_data = g2o_and_time_to_pose_data(result_g2o_file, 
                                                  odom_sparse_all_time_file, robot_id=i)
            pose_data.to_csv(os.path.join(args.output_dir, "offline_rpgo", f"{run}.csv"))
            print(f"Saving {run} pose data to {os.path.join(args.output_dir, 'offline_rpgo', f'{run}.csv')}")

        # Report ATE results
        if args.gt is not None:
            print("ATE results:")
            print("============")
            print(evaluate(result_g2o_file, odom_sparse_all_time_file, {i: args.gt[i] for i in range(len(args.gt))}))