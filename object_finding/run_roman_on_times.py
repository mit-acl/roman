"""Build a ROMAN map from a bag using exactly the frames an existing map used.

The ROS 2 ROMAN node processes whichever frame is current when it finishes the last one, so
its frame times are irregular. A reference map stores those times; this runs ROMAN's
offline mapper (the same code demo/mapping.py uses) on them, so a rerun with different
settings, e.g. another DINOv2 model, sees the same frames.

Example:
    python run_roman_on_times.py --params roman_params/indoor_small_objs --color-names \
        --dino-model facebook/dinov2-with-registers-base \
        --reference-map /path/to/small_objs.roman_map.pkl --out rerun/small_objs_dinov2_reg_color.pkl
"""
import argparse
import os
import pickle
import time

import numpy as np

import segment_color
from roman.map.map import load_roman_map
from roman.map.run import ROMANMapRunner
from roman.params.data_params import DataParams
from roman.params.fastsam_params import FastSAMParams
from roman.params.mapper_params import MapperParams


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--params', required=True, help='Directory with data.yaml, fastsam.yaml, mapper.yaml')
    parser.add_argument('--run', default='indoor', help='Run name in data.yaml')
    parser.add_argument('--reference-map', required=True, help='ROMAN map whose frame times to use')
    parser.add_argument('--out', required=True, help='Output map .pkl')
    parser.add_argument('--dino-model', default=None, help='Override fastsam.yaml dino_model')
    parser.add_argument('--max-frames', type=int, default=None, help='Only process the first N frames (testing)')
    parser.add_argument('--color-names', action='store_true',
                        help='Also accumulate a color-name histogram per segment (segment_color.py); '
                             'saved next to the map as <out>.colors.npz')
    args = parser.parse_args()

    data_params = DataParams.from_yaml(os.path.join(args.params, 'data.yaml'), run=args.run)
    fastsam_params = FastSAMParams.from_yaml(os.path.join(args.params, 'fastsam.yaml'))
    mapper_path = os.path.join(args.params, 'mapper.yaml')
    mapper_params = MapperParams.from_yaml(mapper_path) if os.path.exists(mapper_path) else MapperParams()
    if args.dino_model:
        fastsam_params.dino_model = args.dino_model
    print(f'DINOv2 model: {fastsam_params.dino_model}; min_max_extent: {mapper_params.min_max_extent}')

    if args.color_names:
        segment_color.install_hooks(segment_color.load_w2c())
    times = load_roman_map(os.path.expanduser(args.reference_map)).times[:args.max_frames]
    runner = ROMANMapRunner(data_params=data_params, fastsam_params=fastsam_params,
                            mapper_params=mapper_params, verbose=True)
    print(f'Processing {len(times)} frames from the reference map')
    wall_t0 = time.time()
    for k, t in enumerate(times):
        runner.update(t)
        if (k + 1) % 100 == 0 or k + 1 == len(times):
            rate = (k + 1) / (time.time() - wall_t0)
            print(f'  {k + 1}/{len(times)} frames, {rate:.1f} frames/s, '
                  f'{(len(times) - k - 1) / rate / 60:.1f} min left', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    roman_map = runner.mapper.get_roman_map()
    with open(args.out, 'wb') as f:
        pickle.dump(roman_map, f, -1)
    print(f'Saved {args.out}: {len(roman_map.segments)} segments, {len(roman_map.times)} poses, '
          f'mean {np.mean(runner.processing_times.total_times):.3f} s/frame')
    if args.color_names:
        ids, colors, cnt = segment_color.segment_colors(roman_map.segments)
        color_path = os.path.splitext(args.out)[0] + '.colors.npz'
        np.savez(color_path, segment_id=ids, color_names=colors, n_obs=cnt,
                 names=np.array(segment_color.COLOR_NAMES))
        print(f'Saved {color_path}: color-name histograms for {len(ids)} segments')


if __name__ == '__main__':
    main()
