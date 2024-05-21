import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import pickle
import tqdm
import yaml
import time
import cv2 as cv
import signal
import sys
import open3d as o3d
import logging
import os

from robot_utils.robot_data.img_data import ImgData
from robot_utils.robot_data.pose_data import PoseData
from robot_utils.transform import transform, T_RDFFLU, T_FLURDF
from robot_utils.robot_data.general_data import GeneralData
from robot_utils.robot_data.robot_data import NoDataNearTimeException

from plot_utils import remove_ticks

from img_utils import draw_cylinder

from segment_track.observation import Observation
from segment_track.segment import Segment
from segment_track.tracker import Tracker
from segment_track.fastsam_wrapper import FastSAMWrapper

def draw(t, img, pose, tracker, observations, reprojected_bboxs, ax):
    for axi in ax:
        axi.clear()
        remove_ticks(axi)

    img_fastsam = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_fastsam = np.concatenate([img_fastsam[...,None]]*3, axis=2)

    segment: Segment
    for i, segment in enumerate(tracker.segments + tracker.segment_graveyard):
        # only draw segments seen in the last however many seconds
        if segment.last_seen < t - 50:
            continue
        bbox = segment.reprojected_bbox(pose)
        if bbox is None:
            continue
        if i < len(tracker.segments):
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        img = cv.rectangle(img, np.array([bbox[0][0], bbox[0][1]]).astype(np.int32), 
                    np.array([bbox[1][0], bbox[1][1]]).astype(np.int32), color=color, thickness=2)
        img = cv.putText(img, str(segment.id), (np.array(bbox[0]) + np.array([10., 10.])).astype(np.int32), 
                         cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    matched_masks = []
    for segment in tracker.segments:
        if segment.last_seen == t:
            colored_mask = np.zeros_like(img)
            np.random.seed(segment.id)
            rand_color = np.random.randint(0, 255, 3)
            # print(rand_color)
            matched_masks.append(segment.last_mask)
            try:
                colored_mask = segment.last_mask.astype(np.int32)[..., np.newaxis]*rand_color
            except:
                import ipdb; ipdb.set_trace()
            colored_mask = colored_mask.astype(np.uint8)
            img_fastsam = cv.addWeighted(img_fastsam, 1.0, colored_mask, 0.5, 0)
            for obs in segment.observations[::-1]:
                if obs.time == t:
                    img_fastsam = cv.putText(img_fastsam, str(segment.id), obs.pixel.astype(np.int32), 
                         cv.FONT_HERSHEY_SIMPLEX, 0.5, rand_color.tolist(), 2)
                    break
    
    for obs in observations:
        alread_shown = False
        for mask in matched_masks:
            if np.all(mask == obs.mask):
                alread_shown = True
                break
        if alread_shown:
            continue
        white_mask = obs.mask.astype(np.int32)[..., np.newaxis]*np.ones(3)*255
        white_mask = white_mask.astype(np.uint8)
        img_fastsam = cv.addWeighted(img_fastsam, 1.0, white_mask, 0.5, 0)

    for seg_id, bbox in reprojected_bboxs:
        np.random.seed(seg_id)
        rand_color = np.random.randint(0, 255, 3)
        cv.rectangle(img_fastsam, np.array([bbox[0][0], bbox[0][1]]).astype(np.int32), 
                     np.array([bbox[1][0], bbox[1][1]]).astype(np.int32), color=rand_color.tolist(), thickness=2)
            

    ax[0].imshow(img[...,::-1])
    ax[1].imshow(img_fastsam)
    return


def update(t, img_data, depth_data, pose_data, fastsam, tracker, ax, poses_history):

    try:
        img = img_data.img(t)
        img_depth = depth_data.img(t)
        img_t = img_data.nearest_time(t)
        pose = pose_data.T_WB(img_t)
    except NoDataNearTimeException:
        return
    
    # collect reprojected masks
    reprojected_bboxs = []
    segment: Segment
    for i, segment in enumerate(tracker.segments + tracker.segment_graveyard):
        # only draw segments seen in the last however many seconds
        if segment.last_seen < t - 50:
            continue
        bbox = segment.reprojected_bbox(pose)
        if bbox is not None:
            reprojected_bboxs.append((segment.id, bbox))

    observations = fastsam.run(t, pose, img, img_depth=img_depth)

    if len(observations) > 0:
        tracker.update(t, pose, observations)

    print("t={}, num_obs={}, num_seg={}, num_nurse={}".format(
        t, len(observations), len(tracker.segments), len(tracker.segment_nursery))
    )
    if len(tracker.segments) > 0:
        for seg in tracker.segments:
            print("seg id={}, num_points={}".format(seg.id, seg.num_points))

    if ax is not None:
        draw(t, img, pose, tracker, observations, reprojected_bboxs, ax)

    poses_history.append(pose)

    return
    

def main(args):
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)

    if 'time' in params:
        assert 'relative' in params['time'], "relative must be specified in params"
        assert 't0' in params['time'], "t0 must be specified in params"
        assert 'tf' in params['time'], "tf must be specified in params"

    assert params['img_data']['path'] is not None, "bag must be specified in params"
    assert params['img_data']['img_topic'] is not None, "img_topic must be specified in params"
    assert params['img_data']['cam_info_topic'] is not None, "cam_info_topic must be specified in params"
    if 'depth_compressed_rvl' not in params['img_data']:
        params['img_data']['depth_compressed_rvl'] = False
    assert not (('T_body_cam' in params['pose_data'] or 'T_body_odom' in params['pose_data']) \
        and 'T_postmultiply' in params['pose_data']), "Cannot specify both T_body_cam and T_postmultiply in pose_data"
    assert params['pose_data']['file_type'] != 'bag' or 'topic' in params['pose_data'], \
        "pose topic must be specified in params"
    assert params['pose_data']['time_tol'] is not None, "pose time_tol must be specified in params"

    assert params['fastsam']['weights'] is not None, "weights must be specified in params"
    assert params['fastsam']['imgsz'] is not None, "imgsz must be specified in params"
    if 'device' not in params['fastsam']:
        params['fastsam']['device'] = 'cuda'
    if 'min_mask_len_div' not in params['fastsam']:
        params['fastsam']['min_mask_len_div'] = 30
    if 'max_mask_len_div' not in params['fastsam']:
        params['fastsam']['max_mask_len_div'] = 5
    if 'ignore_people' not in params['fastsam']:
        params['fastsam']['ignore_people'] = False
    if 'erosion_size' not in params['fastsam']:
        params['fastsam']['erosion_size'] = 5
    
    assert params['segment_tracking']['min_iou'] is not None, "min_iou must be specified in params"
    assert params['segment_tracking']['min_sightings'] is not None, "min_sightings must be specified in params"
    assert params['segment_tracking']['pixel_std_dev'] is not None, "max_t_no_sightings must be specified in params"
    assert params['segment_tracking']['max_t_no_sightings'] is not None, "max_t_no_sightings must be specified in params"
    assert params['segment_tracking']['mask_downsample_factor'] is not None, "Mask downsample factor cannot be none!"

    # Setup logging
    # TODO: add support for logfile
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(message)s', 
        datefmt='%m-%d %H:%M:%S', 
        # handlers=logging.StreamHandler()
    )
    
    print("Loading image data...")

    img_file_path = os.path.expanduser(os.path.expandvars(params["img_data"]["path"]))
    if 'time' in params:
        if 'relative' in params['time'] and params['time']['relative']:
            topic_t0 = ImgData.topic_t0(img_file_path, params["img_data"]["img_topic"])
            time_range = [topic_t0 + params['time']['t0'], topic_t0 + params['time']['tf']]
        else:
            time_range = [params['time']['t0'], params['time']['tf']]
    else:
        time_range = None

    print(f"Time range: {time_range}")
    img_data = ImgData(
        data_file=img_file_path,
        file_type='bag',
        topic=params["img_data"]["img_topic"],
        time_tol=.02,
        time_range=time_range
    )
    img_data.extract_params(params['img_data']['cam_info_topic'])

    t0 = img_data.t0
    tf = img_data.tf

    print("Loading depth data for time range {}...".format(time_range))
    depth_data = ImgData(
        data_file=os.path.expanduser(os.path.expandvars(params["img_data"]["path"])),
        file_type='bag',
        topic=params['img_data']['depth_img_topic'],
        time_tol=.02,
        time_range=time_range,
        compressed=params['img_data']['depth_compressed'],
        compressed_rvl=params['img_data']['depth_compressed_rvl'],
    )
    depth_data.extract_params(params['img_data']['depth_cam_info_topic'])

    print("Loading pose data...")
    T_postmultiply = np.eye(4)
    # if 'T_body_odom' in params['pose_data']:
    #     T_postmultiply = np.linalg.inv(np.array(params['pose_data']['T_body_odom']).reshape((4, 4)))
    if 'T_body_cam' in params['pose_data']:
        T_postmultiply = T_postmultiply @ np.array(params['pose_data']['T_body_cam']).reshape((4, 4))
    if 'T_postmultiply' in params['pose_data']:
        if params['pose_data']['T_postmultiply'] == 'T_FLURDF':
            T_postmultiply = T_FLURDF
        elif params['pose_data']['T_postmultiply'] == 'T_RDFFLU':
            T_postmultiply = T_RDFFLU
    
    

    pose_file_path = params['pose_data']['path']
    pose_file_type = params['pose_data']['file_type']
    pose_time_tol = params['pose_data']['time_tol']
    if pose_file_type == 'bag':
        pose_topic = params['pose_data']['topic']
        csv_options = None
    else:
        pose_topic = None
        csv_options = params['pose_data']['csv_options']
        
    pose_data = PoseData(
        data_file=os.path.expanduser(pose_file_path),
        file_type=pose_file_type,
        topic=pose_topic,
        csv_options=csv_options,
        time_tol=pose_time_tol,
        interp=True,
        T_postmultiply=T_postmultiply
    )
    
    print("Setting up FastSAM...")
    fastsam = FastSAMWrapper(
        weights=os.path.expanduser(os.path.expandvars(params['fastsam']['weights'])),
        imgsz=params['fastsam']['imgsz'],
        device=params['fastsam']['device'],
        mask_downsample_factor=params['segment_tracking']['mask_downsample_factor']
    )
    fastsam.setup_rgbd_params(
        depth_cam_params=depth_data.camera_params, 
        max_depth=8,
        depth_scale=1e3,
        voxel_size=0.05,
        erosion_size=params['fastsam']['erosion_size']
    )
    img_area = img_data.camera_params.width * img_data.camera_params.height
    fastsam.setup_filtering(
        ignore_people=params['fastsam']['ignore_people'],
        yolo_det_img_size=(128, 128) if params['fastsam']['ignore_people'] else None,
        allow_tblr_edges=[True, True, True, True],
        area_bounds=[img_area / (params['fastsam']['min_mask_len_div']**2), img_area / (params['fastsam']['max_mask_len_div']**2)]
    )

    print("Setting up segment tracker...")
    tracker = Tracker(
        camera_params=img_data.camera_params,
        pixel_std_dev=params['segment_tracking']['pixel_std_dev'],
        min_iou=params['segment_tracking']['min_iou'],
        min_sightings=params['segment_tracking']['min_sightings'],
        max_t_no_sightings=params['segment_tracking']['max_t_no_sightings'],
        mask_downsample_factor=params['segment_tracking']['mask_downsample_factor']
    )

    print("Running segment tracking! Start time {:.1f}, end time {:.1f}".format(t0, tf))
    wc_t0 = time.time()
    if not args.no_vid:
        fig, ax = plt.subplots(1, 2, dpi=400, layout='tight')
    else:
        fig = None
        ax = None
    poses_history = []
    def update_wrapper(t): 
        update(t, img_data, depth_data, pose_data, fastsam, tracker, ax, poses_history)
        print(f"t: {t - t0:.2f} = {t}")
        # fig.suptitle(f"t: {t - t0:.2f}")

    if not args.no_vid:
        ani = FuncAnimation(fig, update_wrapper, frames=tqdm.tqdm(np.arange(t0, tf, params['segment_tracking']['dt'])), interval=10, repeat=False)
        if not args.output:
            plt.show()
        else:
            video_file = os.path.expanduser(os.path.expandvars(args.output)) + ".mp4"
            writervideo = FFMpegWriter(fps=int(.5*1/params['segment_tracking']['dt']))
            ani.save(video_file, writer=writervideo)          
    else:
        for t in np.arange(t0, tf, params['segment_tracking']['dt']):
            update_wrapper(t)

    print(f"Segment tracking took {time.time() - wc_t0:.2f} seconds")
    print(f"Run duration was {tf - t0:.2f} seconds")
    print(f"Compute per second: {(time.time() - wc_t0) / (tf - t0):.2f}")

    print(f"Number of poses: {len(poses_history)}.")

    if args.output:
        pkl_path = os.path.expanduser(os.path.expandvars(args.output)) + ".pkl"
        pkl_file = open(pkl_path, 'wb')
        # for seg in tracker.segments + tracker.segment_graveyard + tracker.segment_nursery:
        #     for obs in seg.observations:
        #         obs.keypoints = None
        pickle.dump([tracker, poses_history], pkl_file, -1)
        logging.info(f"Saved tracker, poses_history to file: {pkl_path}.")
        pkl_file.close()

    poses_list = []
    pcd_list = []
    for Twb in poses_history:
        pose_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        pose_obj.transform(Twb)
        poses_list.append(pose_obj)
    for seg in tracker.segments + tracker.segment_graveyard:
        seg_points = seg.points
        if seg_points is not None:
            num_pts = seg_points.shape[0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(seg_points)
            rand_color = np.random.uniform(0, 1, size=(1,3))
            rand_color = np.repeat(rand_color, num_pts, axis=0)
            pcd.colors = o3d.utility.Vector3dVector(rand_color)
            pcd_list.append(pcd)
    
    o3d.visualization.draw_geometries(poses_list + pcd_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, help='Path to params file', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to output file', required=False, default=None)
    parser.add_argument('--no-vid', action='store_true', help='Do not show or save video')
    args = parser.parse_args()

    main(args)