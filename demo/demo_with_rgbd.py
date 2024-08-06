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
from os.path import expandvars
from threading import Thread

# from robotdatapy.data.img_data import ImgData
# from robotdatapy.data.pose_data import PoseData
# from robotdatapy.transform import transform, T_RDFFLU, T_FLURDF
# from robotdatapy.data.general_data import GeneralData
# from robotdatapy.data.robot_data import NoDataNearTimeException
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

def draw(t, img, pose, tracker, observations, reprojected_bboxs, viz_bbox=False, viz_mask=False):

    if viz_mask:
        img_fastsam = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_fastsam = np.concatenate([img_fastsam[...,None]]*3, axis=2)

    if viz_bbox:
        segment: Segment
        for i, segment in enumerate(tracker.segments + tracker.inactive_segments + tracker.segment_graveyard):
            # only draw segments seen in the last however many seconds
            if segment.last_seen < t - tracker.segment_graveyard_time - 10:
                continue
            bbox = segment.reprojected_bbox(pose)
            if bbox is None:
                continue
            if i < len(tracker.segments):
                color = (0, 255, 0)
            elif i < len(tracker.segments) + len(tracker.inactive_segments):
                color = (255, 0, 0)
            else:
                color = (180, 0, 180)
            img = cv.rectangle(img, np.array([bbox[0][0], bbox[0][1]]).astype(np.int32), 
                        np.array([bbox[1][0], bbox[1][1]]).astype(np.int32), color=color, thickness=2)
            img = cv.putText(img, str(segment.id), (np.array(bbox[0]) + np.array([10., 10.])).astype(np.int32), 
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if viz_mask:
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
                mass_x, mass_y = np.where(segment.last_mask >= 1)
                img_fastsam = cv.putText(img_fastsam, str(segment.id), (int(np.mean(mass_y)), int(np.mean(mass_x))), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, rand_color.tolist(), 2)
        
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
            
    # concatenate images
    if viz_bbox and viz_mask:
        img_ret = np.concatenate([img, img_fastsam], axis=1)
    elif viz_bbox:
        img_ret = img
    elif viz_mask:
        img_ret = img_fastsam
    return img_ret


def update_fastsam(t, img_data, depth_data, pose_data, fastsam):

    try:
        img = img_data.img(t)
        img_depth = depth_data.img(t)
        img_t = img_data.nearest_time(t)
        pose = pose_data.T_WB(img_t)
    except NoDataNearTimeException:
        return None, None, None
    
    observations = fastsam.run(t, pose, img, img_depth=img_depth)
    return observations, pose, img

def update_segment_track(t, observations, pose, img, tracker, 
                         poses_history, times_history, viz_bbox=False, viz_mask=False): 

    # collect reprojected masks
    reprojected_bboxs = []
    img_ret = None
    if viz_mask:
        segment: Segment
        for i, segment in enumerate(tracker.segments + tracker.inactive_segments):
            bbox = segment.reprojected_bbox(pose)
            if bbox is not None:
                reprojected_bboxs.append((segment.id, bbox))

    if len(observations) > 0:
        tracker.update(t, pose, observations)

    print("t={}, num_obs={}, num_seg={}, num_nurse={}".format(
        t, len(observations), len(tracker.segments), len(tracker.segment_nursery))
    )
    if len(tracker.segments) > 0:
        for seg in tracker.segments:
            print("seg id={}, num_points={}".format(seg.id, seg.num_points))

    if viz_bbox or viz_mask:
        img_ret = draw(t, img, pose, tracker, observations, reprojected_bboxs, viz_bbox, viz_mask)

    poses_history.append(pose)
    times_history.append(t)

    return img_ret
    

def main(args):
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)

    # TODO: this is getting really messy. Need a way to clean this up.
    if 'time' in params:
        assert 'relative' in params['time'], "relative must be specified in params"
        assert 't0' in params['time'], "t0 must be specified in params"
        assert 'tf' in params['time'], "tf must be specified in params"

    # use_kitti = params['use_kitti']
    try:
        use_kitti = params['use_kitti']
    except KeyError:
        use_kitti = False

    if not use_kitti:
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

    # FASTSAM PARAMS
    if 'device' not in params['fastsam']:
        params['fastsam']['device'] = 'cuda'
    if 'min_mask_len_div' not in params['fastsam']:
        params['fastsam']['min_mask_len_div'] = 30
    if 'max_mask_len_div' not in params['fastsam']:
        params['fastsam']['max_mask_len_div'] = 3
    if 'ignore_people' not in params['fastsam']:
        params['fastsam']['ignore_people'] = False
    if 'erosion_size' not in params['fastsam']:
        params['fastsam']['erosion_size'] = 3
    if 'max_depth' not in params['fastsam']:
        params['fastsam']['max_depth'] = 8.0
    if 'ignore_labels' not in params['fastsam']:
        params['fastsam']['ignore_labels'] = []
    if 'use_keep_labels' not in params['fastsam']:
        params['fastsam']['use_keep_labels'] = False
    if 'keep_labels' not in params['fastsam']:
        params['fastsam']['keep_labels'] = []
    if 'keep_labels_option' not in params['fastsam']:
        params['fastsam']['keep_labels_option'] = None
    if 'plane_filter_params' not in params['fastsam']:
        params['fastsam']['plane_filter_params'] = np.array([3.0, 1.0, 0.2])
    if 'yolo' not in params:
        params['yolo'] = {'imgsz': params['fastsam']['imgsz']}
        
    # IMG_DATA PARAMS TODO: cleanup
    if 'depth_scale' not in params['img_data']:
        params['img_data']['depth_scale'] = 1e3
    
    
    assert params['segment_tracking']['min_iou'] is not None, "min_iou must be specified in params"
    assert params['segment_tracking']['min_sightings'] is not None, "min_sightings must be specified in params"
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

    if use_kitti:
        time_range = [params['time']['t0'], params['time']['tf']]
    else:
        img_file_path = os.path.expanduser(expandvars(params["img_data"]["path"]))
        if 'time' in params:
            if 'relative' in params['time'] and params['time']['relative']:
                topic_t0 = ImgData.topic_t0(img_file_path, expandvars(params["img_data"]["img_topic"]))
                time_range = [topic_t0 + params['time']['t0'], topic_t0 + params['time']['tf']]
            else:
                time_range = [params['time']['t0'], params['time']['tf']]
        else:
            time_range = None

    print(f"Time range: {time_range}")
    if use_kitti:
        img_data = ImgData.from_kitti(params['img_data']['base_path'], 'rgb')
        img_data.extract_params()
    else:
        img_data = ImgData.from_bag(
            path=img_file_path,
            topic=expandvars(params["img_data"]["img_topic"]),
            time_tol=.02,
            time_range=time_range
        )
        img_data.extract_params(expandvars(params['img_data']['cam_info_topic']))

    t0 = img_data.t0
    tf = img_data.tf

    print("Loading depth data for time range {}...".format(time_range))
    if use_kitti:
        depth_data = ImgData.from_kitti(params['img_data']['base_path'], 'depth')
        depth_data.extract_params()
    else:
        depth_data = ImgData.from_bag(
            path=img_file_path,
            topic=expandvars(params['img_data']['depth_img_topic']),
            time_tol=.02,
            time_range=time_range,
            compressed=params['img_data']['depth_compressed'],
            compressed_rvl=params['img_data']['depth_compressed_rvl'],
        )
        depth_data.extract_params(expandvars(params['img_data']['depth_cam_info_topic']))

    print("Loading pose data...")
    if not use_kitti:
        pose_file_path = params['pose_data']['path']
        pose_file_type = params['pose_data']['file_type']
        pose_time_tol = params['pose_data']['time_tol']
        if pose_file_type == 'bag':
            pose_topic = expandvars(params['pose_data']['topic'])
            csv_options = None
        else:
            pose_topic = None
            csv_options = params['pose_data']['csv_options']
        
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
    if 'tf_to_cam' in params['pose_data']:
        T_postmultiply = PoseData.static_tf_from_bag(
            os.path.expanduser(img_file_path), 
            expandvars(params['pose_data']['tf_to_cam']['parent']), 
            expandvars(params['pose_data']['tf_to_cam']['child'])
        )
    
    if use_kitti:
        pose_data = PoseData.from_kitti(params['pose_data']['base_path'])
    elif pose_file_type == 'bag':
        pose_data = PoseData.from_bag(
            path=os.path.expanduser(pose_file_path),
            topic=expandvars(pose_topic),
            time_tol=pose_time_tol,
            interp=True,
            T_postmultiply=T_postmultiply
        )
    else:
        pose_data = PoseData.from_csv(
            path=os.path.expanduser(pose_file_path),
            csv_options=csv_options,
            time_tol=pose_time_tol,
            interp=True,
            T_postmultiply=T_postmultiply
        )
    
    ### Set up FastSAM
    print("Setting up FastSAM...")
    fastsam = FastSAMWrapper(
        weights=os.path.expanduser(expandvars(params['fastsam']['weights'])),
        imgsz=params['fastsam']['imgsz'],
        device=params['fastsam']['device'],
        mask_downsample_factor=params['segment_tracking']['mask_downsample_factor']
    )
    fastsam.setup_rgbd_params(
        depth_cam_params=depth_data.camera_params, 
        max_depth=params['fastsam']['max_depth'],
        depth_scale=params['img_data']['depth_scale'],
        voxel_size=0.05,
        erosion_size=params['fastsam']['erosion_size'],
        plane_filter_params=params['fastsam']['plane_filter_params']
    )
    img_area = img_data.camera_params.width * img_data.camera_params.height
    fastsam.setup_filtering(
        ignore_labels=params['fastsam']['ignore_labels'],
        use_keep_labels=params['fastsam']['use_keep_labels'],
        keep_labels=params['fastsam']['keep_labels'],
        keep_labels_option=params['fastsam']['keep_labels_option'],
        yolo_det_img_size=params['yolo']['imgsz'],
        allow_tblr_edges=[True, True, True, True],
        area_bounds=[img_area / (params['fastsam']['min_mask_len_div']**2), img_area / (params['fastsam']['max_mask_len_div']**2)],
        clip_embedding=True,
        clip_model='ViT-B/32',
        device='cuda'
    )

    print("Setting up segment tracker...")
    tracker = Tracker(
        camera_params=img_data.camera_params,
        min_iou=params['segment_tracking']['min_iou'],
        min_sightings=params['segment_tracking']['min_sightings'],
        max_t_no_sightings=params['segment_tracking']['max_t_no_sightings'],
        mask_downsample_factor=params['segment_tracking']['mask_downsample_factor']
    )

    print("Running segment tracking! Start time {:.1f}, end time {:.1f}".format(t0, tf))
    wc_t0 = time.time()
    poses_history = []
    times_history = []


    if not args.no_vid:
        fc = cv.VideoWriter_fourcc(*"mp4v")
        video_file = os.path.expanduser(expandvars(args.output)) + ".mp4"
        fps = int(np.max([1., args.vid_rate*1/params['segment_tracking']['dt']]))
        video = cv.VideoWriter(video_file, fc, fps, 
                               (img_data.camera_params.width*(2 if not args.vid_bbox_only else 1), 
                                img_data.camera_params.height))

    def update_wrapper(t): 
        print(f"t: {t - t0:.2f} = {t}")
        observations, pose, img = update_fastsam(t, img_data, depth_data, pose_data, fastsam)
        if observations is not None and pose is not None and img is not None:
            return update_segment_track(t, observations, pose, img, tracker, poses_history, 
                                        times_history, viz_bbox=not args.no_vid, 
                                        viz_mask=not args.no_vid and not args.vid_bbox_only)

    for t in np.arange(t0, tf, params['segment_tracking']['dt']):
        img_t = update_wrapper(t)
        if not args.no_vid and img_t is not None:
            video.write(img_t)
            
        # observations, reprojected_bboxs, pose, img = None, None, None, None
        # new_observations, new_reprojected_bboxs, new_pose, new_img = None, None, None, None
        # def fastsam_update_wrapper(t, img_data, depth_data, pose_data, fastsam, tracker, results):
        #     observations, reprojected_bboxs, pose, img = update_fastsam(t, img_data, depth_data, pose_data, fastsam, tracker)
        #     results[0] = observations
        #     results[1] = reprojected_bboxs
        #     results[2] = pose
        #     results[3] = img
        # results = [0, 0, 0, 0]
        # fastsam_update_wrapper(t0, img_data, depth_data, pose_data, fastsam, tracker, results)
        # new_observations, new_reprojected_bboxs, new_pose, new_img = results
        # # Start here: figure out how to do arguments in threading: https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread
        # for t in np.arange(t0, tf, params['segment_tracking']['dt']):
        #     print(f"t: {t - t0:.2f} = {t}")
        #     observations = new_observations
        #     reprojected_bboxs = new_reprojected_bboxs
        #     pose = new_pose
        #     img = new_img

        #     run_seg_track_t = observations is not None and reprojected_bboxs is not None and pose is not None and img is not None
        #     run_fastsam_tp = t + params['segment_tracking']['dt'] < tf

        #     if run_seg_track_t:
        #         t1 = Thread(target = update_segment_track(t, observations, reprojected_bboxs, pose, img, tracker, ax, poses_history))
        #         t1.start()
        #     if run_fastsam_tp:
        #         t2 = Thread(fastsam_update_wrapper(t+params['segment_tracking']['dt'], img_data, depth_data, pose_data, fastsam, tracker, results))
        #         t2.start()
        #     if run_seg_track_t:
        #         t1.join()
        #     if run_fastsam_tp:
        #         t2.join()
        #         new_observations, new_reprojected_bboxs, new_pose, new_img = results
        #     else: # last iteration
        #         break
    if not args.no_vid:
        video.release()
        cv.destroyAllWindows()

    print(f"Segment tracking took {time.time() - wc_t0:.2f} seconds")
    print(f"Run duration was {tf - t0:.2f} seconds")
    print(f"Compute per second: {(time.time() - wc_t0) / (tf - t0):.2f}")

    print(f"Number of poses: {len(poses_history)}.")

    if args.output:
        pkl_path = os.path.expanduser(expandvars(args.output)) + ".pkl"
        pkl_file = open(pkl_path, 'wb')
        tracker.make_pickle_compatible()
        pickle.dump([tracker, poses_history, times_history], pkl_file, -1)
        logging.info(f"Saved tracker, poses_history to file: {pkl_path}.")
        pkl_file.close()

    if not args.no_o3d:
        poses_list = []
        pcd_list = []
        for Twb in poses_history:
            pose_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            pose_obj.transform(Twb)
            poses_list.append(pose_obj)
        for seg in tracker.segments + tracker.inactive_segments + tracker.segment_graveyard:
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
    parser.add_argument('--no-o3d', action='store_true', help='Do not show o3d visualization')
    parser.add_argument('--vid-rate', type=float, help='Video playback rate', default=1.0)
    parser.add_argument('--vid-bbox-only', action='store_true', help='Only show bounding boxes in video')
    args = parser.parse_args()

    main(args)