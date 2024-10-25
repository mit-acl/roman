import argparse

from roman.align.params import SubmapAlignInputOutput, SubmapAlignParams
from roman.align.submap_align import submap_align

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ROMAN submap alignment/loop closure module')
    parser.add_argument('-i', '--input', type=str, nargs=2, help='Input ROMAN map pkl files', required=True)
    parser.add_argument('-g', '--gt', type=str, nargs=2, help='Ground truth pose files (yaml PoseData loading info)')
    parser.add_argument('-o', '--output-dir', type=str, help='Output directory', required=True)
    parser.add_argument('-n', '--run-name', type=str, help='Run name for file prefix', default='align')
    parser.add_argument('-p', '--params', type=str, default=None, help='Path to params yaml file')
    parser.add_argument('-s', '--single-robot', action='store_true', help='Single robot loop closures')
    args = parser.parse_args()

    if args.params:
        assert False, 'params file not supported yet'

    sm_params = SubmapAlignParams(single_robot_lc=args.single_robot)
    sm_io = SubmapAlignInputOutput(
        inputs=args.input,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    if args.gt:
        sm_io.input_gt_pose_yaml = args.gt

    submap_align(sm_params=sm_params, sm_io=sm_io)