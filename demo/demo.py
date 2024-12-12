import numpy as np
import matplotlib.pyplot as plt

import argparse
from typing import List
import os
import yaml

from roman.align.params import SubmapAlignInputOutput, SubmapAlignParams
from roman.align.submap_align import submap_align
from roman.offline_rpgo.extract_odom_g2o import roman_map_pkl_to_g2o
from roman.offline_rpgo.g2o_file_fusion import create_config, g2o_file_fusion
from roman.offline_rpgo.combine_loop_closures import combine_loop_closures
from roman.offline_rpgo.plot_g2o import plot_g2o, DEFAULT_TRAJECTORY_COLORS, G2OPlotParams
from roman.offline_rpgo.g2o_and_time_to_pose_data import g2o_and_time_to_pose_data
from roman.offline_rpgo.evaluate import evaluate
from roman.offline_rpgo.edit_g2o_edge_information import edit_g2o_edge_information
from roman.offline_rpgo.params import OfflineRPGOParams

import mapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runs', type=str, nargs="+", 
                        help='Run names. Individual map files will be saved using these names.')
    parser.add_argument('-p', '--params', default=None, type=str, help='Path to params directory.' +
                        ' Params can include mapping.yaml with mapping parameters. gt_pose.yaml with ground' +
                        ' truth parameters for evaluation. submap_align.yaml for loop closure parameters.' +
                        ' offline_rpgo.yaml for pose graph optimization parameters. Only mapping.yaml is required,' +
                        ' although a list of mapping params can be given with --mapping-params instead.' +
                        ' When loading these files, --run-env will be set with the run name.', required=False)
    parser.add_argument('--mapping-params', nargs='+', default=None, type=str, required=False,
                        help='List of paths to mapping parameters. Must match number of runs. Overrides --params.')
    parser.add_argument('-o', '--output-dir', type=str, help='Path to output directory', required=True, default=None)
    parser.add_argument('-e', '--run-env', type=str, help='This argument will be set as an environment variable for each run.' +
                        'e.g, "-r run1 run2 --run-env NAME" will set the environment variable NAME=run1 and NAME=run2')
    
    parser.add_argument('-m', '--viz-map', action='store_true', help='Visualize map')
    parser.add_argument('-v', '--viz-observations', action='store_true', help='Visualize observations')
    parser.add_argument('-3', '--viz-3d', action='store_true', help='Visualize 3D')
    parser.add_argument('--vid-rate', type=float, help='Video playback rate', default=1.0)
    parser.add_argument('-d', '--save-img-data', action='store_true', help='Save video frames as ImgData class')
    parser.add_argument('-s', '--sparse-pgo', action='store_true', help='Use sparse pose graph optimization')
    parser.add_argument('-n', '--num-req-assoc', type=int, help='Number of required associations', default=4)
    # parser.add_argument('--set-env-vars', type=str)
    parser.add_argument('--max-time', type=float, default=None, help='If the input data is too large, this allows a maximum time' +
                        'to be set, such that if the mapping will be chunked into max_time increments and fused together')

    parser.add_argument('--skip-map', action='store_true', help='Skip mapping')
    parser.add_argument('--skip-align', action='store_true', help='Skip alignment')
    parser.add_argument('--skip-rpgo', action='store_true', help='Skip robust pose graph optimization')
    parser.add_argument('--skip-indices', type=int, nargs='+', help='Skip specific runs in mapping and alignment')

    args = parser.parse_args()

    # setup params
    assert args.params is not None or args.mapping_params is not None, \
        "Either --params or --mapping-params must be provided."
    if args.mapping_params:
        assert len(args.mapping_params) == len(args.runs), \
            "Number of mapping params must match number of runs."
    has_mapping_params_list = args.mapping_params is not None
    has_params_dir = args.params is not None
    if has_mapping_params_list:
        mapping_params_list = args.mapping_params
    if has_params_dir:
        mapping_params_list = [os.path.join(args.params, f"mapping.yaml") for run in args.runs]
        params_dir = args.params
        submap_align_params_path = os.path.join(args.params, f"submap_align.yaml")
        submap_align_params = SubmapAlignParams.from_yaml(submap_align_params_path) \
            if os.path.exists(submap_align_params_path) else SubmapAlignParams()
        offline_rpgo_params_path = os.path.join(args.params, f"offline_rpgo.yaml")
        offline_rpgo_params = OfflineRPGOParams.from_yaml(offline_rpgo_params_path) \
            if os.path.exists(os.path.join(args.params, "offline_rpgo.yaml")) else OfflineRPGOParams()
            
    else:
        submap_align_params = SubmapAlignParams()
        offline_rpgo_params = OfflineRPGOParams()

    # ground truth pose files
    if has_params_dir and os.path.exists(os.path.join(params_dir, "gt_pose.yaml")):
        has_gt = True
        gt_files = [os.path.join(params_dir, "gt_pose.yaml") for _ in range(len(args.runs))]
    else:
        has_gt = False
        gt_files = [None for _ in range(len(args.runs))]

    # create output directories
    os.makedirs(os.path.join(args.output_dir, "map"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "align"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "offline_rpgo"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "offline_rpgo/sparse"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "offline_rpgo/dense"), exist_ok=True)
    
    if not args.skip_map:
        
        for i, run in enumerate(args.runs):
            if args.skip_indices and i in args.skip_indices:
                continue
                
            # mkdir $output_dir/map
            args.output = os.path.join(args.output_dir, "map", f"{run}")
            args.params = mapping_params_list[i]

            # shell: export RUN=run
            if args.run_env is not None:
                os.environ[args.run_env] = run
            
            print(f"Mapping: {run}")
            mapping.mapping(args)
        
    if not args.skip_align:
        # TODO: support ground truth pose file for validation
            
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
                    input_gt_pose_yaml=[gt_files[i], gt_files[j]],
                    robot_names=[args.runs[i], args.runs[j]],
                    robot_env=args.run_env,
                )
                submap_align_params.single_robot_lc = (i == j)
                submap_align(sm_params=submap_align_params, sm_io=sm_io)
                       
    if not args.skip_rpgo:
        min_keyframe_dist = 0.01 if not args.sparse_pgo else 2.0
        # Create g2o files for odometry
        for i, run in enumerate(args.runs):
            roman_map_pkl_to_g2o(
                pkl_file=os.path.join(args.output_dir, "map", f"{run}.pkl"),
                g2o_file=os.path.join(args.output_dir, "offline_rpgo/sparse", f"{run}.g2o"),
                time_file=os.path.join(args.output_dir, "offline_rpgo/sparse", f"{run}.time.txt"),
                robot_id=i,
                min_keyframe_dist=min_keyframe_dist,
                t_std=offline_rpgo_params.odom_t_std,
                r_std=offline_rpgo_params.odom_r_std,
                verbose=True
            )
            
            # create dense g2o file
            roman_map_pkl_to_g2o(
                pkl_file=os.path.join(args.output_dir, "map", f"{run}.pkl"),
                g2o_file=os.path.join(args.output_dir, "offline_rpgo/dense", f"{run}.g2o"),
                time_file=os.path.join(args.output_dir, "offline_rpgo/dense", f"{run}.time.txt"),
                robot_id=i,
                min_keyframe_dist=None,
                t_std=offline_rpgo_params.odom_t_std,
                r_std=offline_rpgo_params.odom_r_std,
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
        if args.sparse_pgo:
            final_g2o_file = os.path.join(args.output_dir, "offline_rpgo", "odom_and_lc.g2o")
            combine_loop_closures(
                g2o_reference=odom_sparse_all_g2o_file, 
                g2o_extra_lc=dense_g2o_file, 
                vertex_times_reference=odom_sparse_all_time_file,
                vertex_times_extra_lc=odom_dense_all_time_file,
                output_file=final_g2o_file,
            )
        else:
            final_g2o_file = dense_g2o_file
        
        # change lc covar
        with open(os.path.expanduser(final_g2o_file), 'r') as f:
            g2o_lines = f.readlines()
        g2o_lines = edit_g2o_edge_information(g2o_lines, offline_rpgo_params.lc_t_std, 
                                              offline_rpgo_params.lc_r_std, loop_closures=True)
        with open(os.path.expanduser(final_g2o_file), 'w') as f:
            for line in g2o_lines:
                f.write(line + '\n')
            f.close()
            
        # run kimera centralized robust pose graph optimization
        result_g2o_file = os.path.join(args.output_dir, "offline_rpgo", "result.g2o")
        ros_launch_command = f"roslaunch kimera_centralized_pgmo offline_g2o_solver.launch \
            g2o_file:={final_g2o_file} \
            output_path:={os.path.join(args.output_dir, 'offline_rpgo')}"
        roman_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        rpgo_read_g2o_executable = \
            f"{roman_path}/dependencies/Kimera-RPGO/build/RpgoReadG2o"
        rpgo_command = f"{rpgo_read_g2o_executable} 3d {final_g2o_file}" \
            + f" -1.0 -1.0 0.9 {os.path.join(args.output_dir, 'offline_rpgo')} v"
        os.system(rpgo_command)
        # os.system(ros_launch_command)
        
        # plot results
        g2o_symbol_to_name = {chr(97 + i): args.runs[i] for i in range(len(args.runs))}
        g2o_plot_params = G2OPlotParams()
        fig, ax = plt.subplots(3, 1, figsize=(5,10))
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
                                                  odom_sparse_all_time_file if args.sparse_pgo else odom_dense_all_time_file, 
                                                  robot_id=i)
            pose_data.to_csv(os.path.join(args.output_dir, "offline_rpgo", f"{run}.csv"))
            print(f"Saving {run} pose data to {os.path.join(args.output_dir, 'offline_rpgo', f'{run}.csv')}")

        # Report ATE results
        if has_gt:
            print("ATE results:")
            print("============")
            print(evaluate(
                result_g2o_file, 
                odom_sparse_all_time_file  if args.sparse_pgo else odom_dense_all_time_file, 
                {i: gt_files[i] for i in range(len(gt_files))},
                {i: args.runs[i] for i in range(len(args.runs))},
                args.run_env,
                output_dir=args.output_dir
            ))