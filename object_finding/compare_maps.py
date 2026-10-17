"""Match the segments of two ROMAN maps of the same run by 3D overlap and compare them.

Both maps must be in the same odometry frame (e.g. a rerun of the same bag). Segments are
voxelized and matched one-to-one by voxel IoU (Hungarian assignment). Reports how many
segments match, how far apart matched centroids are and, when both maps used the same
descriptor model, how similar their DINOv2 descriptors are. With --labels, also reports
what happened to each labeled segment and can write the labels onto the new map's
segments (--out-labels), for maps built with a different model.

Example:
    python compare_maps.py --ref /path/to/small_objs.roman_map.pkl --new rerun/small_objs_dinov2_reg_color.pkl \
        --labels labels_indoor_small_objs/small_objs_labels.csv --out-labels rerun/small_objs_dinov2_reg_labels.csv
"""
import argparse
import csv
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from roman.map.map import load_roman_map


def voxels(points, size):
    return set(map(tuple, np.floor(points / size).astype(np.int64)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ref', required=True, help='Reference map .pkl (the labeled one)')
    parser.add_argument('--new', required=True, help='New map .pkl')
    parser.add_argument('--labels', default=None, help='Label CSV for the reference map')
    parser.add_argument('--out-labels', default=None, help='Write the labels transferred to the new map here')
    parser.add_argument('--voxel', type=float, default=0.05, help='Voxel size for overlap, m')
    parser.add_argument('--min-iou', type=float, default=0.3, help='Minimum IoU to count as a match')
    args = parser.parse_args()

    ref = [s for s in load_roman_map(os.path.expanduser(args.ref)).segments if s.semantic_descriptor is not None]
    new = [s for s in load_roman_map(os.path.expanduser(args.new)).segments if s.semantic_descriptor is not None]
    print(f'reference: {len(ref)} segments, new: {len(new)} segments')

    ref_vox = [voxels(s.points, args.voxel) for s in ref]
    new_vox = [voxels(s.points, args.voxel) for s in new]
    ref_c = np.array([s.points.mean(axis=0) for s in ref])
    new_c = np.array([s.points.mean(axis=0) for s in new])
    iou = np.zeros((len(ref), len(new)))
    for i in range(len(ref)):
        for j in np.nonzero(np.linalg.norm(new_c - ref_c[i], axis=1) < 3.0)[0]:  # only nearby pairs
            inter = len(ref_vox[i] & new_vox[j])
            if inter:
                iou[i, j] = inter / len(ref_vox[i] | new_vox[j])
    rows, cols = linear_sum_assignment(-iou)
    match = {r: c for r, c in zip(rows, cols) if iou[r, c] >= args.min_iou}

    ious = np.array([iou[r, c] for r, c in match.items()])
    dist = np.array([np.linalg.norm(ref_c[r] - new_c[c]) for r, c in match.items()])
    cos = np.array([ref[r].semantic_descriptor.ravel() @ new[c].semantic_descriptor.ravel() /
                    np.linalg.norm(ref[r].semantic_descriptor) / np.linalg.norm(new[c].semantic_descriptor)
                    for r, c in match.items()])
    same_id = np.mean([ref[r].id == new[c].id for r, c in match.items()])
    print(f'matched (IoU >= {args.min_iou}): {len(match)} of {len(ref)} reference segments '
          f'({len(new) - len(match)} new segments unmatched)')
    print(f'  IoU median {np.median(ious):.2f}, 10th pct {np.percentile(ious, 10):.2f}')
    print(f'  centroid distance median {100 * np.median(dist):.1f} cm, 90th pct {100 * np.percentile(dist, 90):.1f} cm')
    print(f'  descriptor cosine median {np.median(cos):.3f}, 10th pct {np.percentile(cos, 10):.3f} '
          f'(only meaningful if both maps used the same DINOv2 model)')
    print(f'  same segment id: {100 * same_id:.0f}% of matches')

    if not args.labels:
        return
    with open(args.labels, newline='') as f:
        label_rows = list(csv.DictReader(f))
    by_id = {r['segment_id']: r for r in label_rows}
    print('\nLabeled (non-background) reference segments:')
    for r, s in enumerate(ref):
        lab = by_id.get(str(s.id))
        if not lab or lab['label'] in ('', 'background'):
            continue
        if r in match:
            c = match[r]
            print(f'  {s.id:>5} {lab["label"]!r}: new segment {new[c].id}, IoU {iou[r, c]:.2f}, '
                  f'centroid {100 * np.linalg.norm(ref_c[r] - new_c[c]):.0f} cm, descriptor cosine '
                  f'{cos[list(match).index(r)]:.3f}')
        else:
            best = int(np.argmax(iou[r]))
            print(f'  {s.id:>5} {lab["label"]!r}: NO MATCH (best IoU {iou[r, best]:.2f} with new segment {new[best].id})')

    if args.out_labels:
        out = []
        for r, c in match.items():
            lab = by_id.get(str(ref[r].id))
            if lab and lab['label']:
                out.append({'run': lab['run'], 'segment_id': new[c].id, 'ref_segment_id': ref[r].id,
                            'iou': f'{iou[r, c]:.2f}', 'object_id': lab['object_id'], 'label': lab['label']})
        with open(args.out_labels, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            writer.writeheader()
            writer.writerows(out)
        print(f'\nWrote {len(out)} transferred labels to {args.out_labels}')


if __name__ == '__main__':
    main()
