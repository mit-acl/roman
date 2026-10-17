"""Save image crops of every object in a ROMAN map and a CSV template for labeling them.

For each segment, the observations with the largest reprojected bounding boxes are
looked up in the original image data and saved side by side as crops/<run>_<id>.jpg.
The full frame of the largest view, with the object boxed, is saved as
context/<run>_<id>.jpg so you can see what the object is. Object pixels (the segment's
3D points projected into the image) are outlined. Rerunning keeps any labels already
entered in an existing CSV.

Images come either from a ROMAN params directory (--params, using its data.yaml) or
straight from a bag (--bag and --topic).

Examples:
    python make_crops.py --params ~/Projects/ACL/roman/params/demo --run sparkal1 \
        --map ~/Projects/ACL/roman/demo_output/map/sparkal1.pkl --out labels_demo
    python make_crops.py --bag /path/to/bag_dir --topic /robot/camera/image_raw --run indoor \
        --map /path/to/map.pkl --out labels_indoor
"""
import argparse
import csv
import os

import cv2 as cv
import numpy as np

from roman.map.map import load_roman_map
from roman.params.data_params import DataParams
from robotdatapy.data import ImgData

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--params', help='ROMAN params directory (uses its data.yaml)')
    parser.add_argument('--bag', help='ROS1 .bag file or ROS2 bag directory, instead of --params')
    parser.add_argument('--topic', help='Image topic in --bag')
    parser.add_argument('--compressed', action='store_true', help='--topic is a CompressedImage topic')
    parser.add_argument('--run', required=True, help='Run name, used in file names (e.g. sparkal1)')
    parser.add_argument('--map', required=True, help='ROMAN map .pkl for this run')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--views', type=int, default=3, help='Crops per object')
    parser.add_argument('--min-px', type=int, default=20, help='Skip views whose box is smaller than this')
    args = parser.parse_args()

    if args.bag:
        if not args.topic:
            parser.error('--bag needs --topic')
        img_data = ImgData.from_bag(os.path.expanduser(args.bag), args.topic, compressed=args.compressed)
    elif args.params:
        data_params_path = os.path.join(os.path.expanduser(args.params), 'data.yaml')
        data_params = DataParams.from_yaml(data_params_path, run=args.run)
        if data_params.run_env is not None:
            os.environ[data_params.run_env] = args.run  # data.yaml paths use e.g. ${ROBOT}
        img_data = data_params.load_img_data()
    else:
        parser.error('give either --params or --bag/--topic')
    roman_map = load_roman_map(os.path.expanduser(args.map))

    crop_dir = os.path.join(args.out, 'crops')
    context_dir = os.path.join(args.out, 'context')
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(context_dir, exist_ok=True)
    csv_path = os.path.join(args.out, f'{args.run}_labels.csv')
    rows = []

    for seg in roman_map.segments:
        if seg.semantic_descriptor is None:
            continue
        # one candidate per distinct observation time, ranked by reprojected box area
        candidates = {}
        for obs in seg.observations:
            bbox = seg.reprojected_bbox(obs.pose)
            if bbox is None:
                continue
            (x0, y0), (x1, y1) = bbox
            if min(x1 - x0, y1 - y0) < args.min_px:
                continue
            area = (x1 - x0) * (y1 - y0)
            if obs.time not in candidates or area > candidates[obs.time][0]:
                candidates[obs.time] = (area, obs.pose.copy(), bbox)
        best = sorted(candidates.items(), key=lambda kv: -kv[1][0])[:args.views]

        tiles, context = [], None
        for t, (_, pose, ((x0, y0), (x1, y1))) in best:
            try:
                img = img_data.img(t)
            except Exception:
                continue
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]  # drop alpha, e.g. ZED images are BGRA
            seg._pixels = None  # force reprojection for this pose
            pixels = seg._pixels_2d(pose)
            if pixels is None:
                continue
            px = np.round(pixels).astype(int)
            px[:, 0] = np.clip(px[:, 0], 0, img.shape[1] - 1)
            px[:, 1] = np.clip(px[:, 1], 0, img.shape[0] - 1)

            pad = int(0.15 * max(x1 - x0, y1 - y0))
            cx0, cy0 = max(x0 - pad, 0), max(y0 - pad, 0)
            cx1, cy1 = min(x1 + pad, img.shape[1]), min(y1 + pad, img.shape[0])
            tile = img[cy0:cy1, cx0:cx1].copy()
            hull = cv.convexHull((px - [cx0, cy0]).astype(np.int32))
            cv.polylines(tile, [hull], True, (255, 255, 255), 1)
            scale = 200 / tile.shape[0]
            tiles.append(cv.resize(tile, (max(int(tile.shape[1] * scale), 1), 200)))

            if context is None:  # full frame of the largest view, object boxed
                context = img.copy()
                cv.rectangle(context, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (0, 255, 255), 2)
                cv.polylines(context, [cv.convexHull(px.astype(np.int32))], True, (255, 255, 255), 1)

        if not tiles:
            continue
        sheet = np.hstack(tiles)
        cv.putText(sheet, f'{args.run} id {seg.id}', (5, 18), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        crop_name = f'{args.run}_{seg.id}.jpg'
        cv.imwrite(os.path.join(crop_dir, crop_name), sheet)
        cv.imwrite(os.path.join(context_dir, crop_name), context)
        rows.append({
            'run': args.run, 'segment_id': seg.id, 'crop': f'crops/{crop_name}',
            'n_views': seg.semantic_descriptor_cnt,
            'label': '', 'object_id': '',
        })

    # keep labels already entered in an existing CSV (read now, just before overwriting)
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            old = {(r['run'], r['segment_id']): r for r in csv.DictReader(f)}
        kept = 0
        for r in rows:
            prev = old.get((r['run'], str(r['segment_id'])))
            if prev:
                # older CSVs split labels into adjective/noun or color/category columns
                old = [prev.get(c, '') for c in ['adjective', 'noun', 'color', 'category']]
                r['label'] = prev.get('label') or ' '.join(x for x in old if x)
                r['object_id'] = prev.get('object_id', '')
                kept += any(r[k] for k in ['label', 'object_id'])
        print(f'Kept existing labels for {kept} objects from {csv_path}')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} objects: crops in {crop_dir}, labels template {csv_path}')
    print('Label with label_server.py: any text per object (e.g. "orange tennis ball" or "background").')
    print('Optional "object_id" if one physical object appears as several segments or runs.')


if __name__ == '__main__':
    main()
