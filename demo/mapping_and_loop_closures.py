import argparse
from typing import List
import os

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
    parser.add_argument('--set-env-vars', type=str)
    args = parser.parse_args()

    assert args.params is not None or args.params_list is not None, "Either --params or --params-list must be provided."
    
    if args.params_list:
        assert len(args.params_list) == len(args.runs), "Number of params files must match number of runs."
    
    args.viz_open3d = False
        
    for i, run in enumerate(args.runs):
        if args.params_list:
            args.params = args.params_list[i]
        # mkdir $output_dir/map
        os.makedirs(os.path.join(args.output_dir, "map"), exist_ok=True)
        args.output = os.path.join(args.output_dir, "map", f"{run}")
            
        # shell: export RUN=run
        os.environ['RUN'] = run
        
        demo.main(args)