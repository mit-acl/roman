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
from robot_utils.transform import transform
from robot_utils.robot_data.general_data import GeneralData

from plot_utils import remove_ticks

from img_utils import draw_cylinder

from segment_track.observation import Observation
from segment_track.segment import Segment
from segment_track.tracker import Tracker
from segment_track.fastsam_wrapper import FastSAMWrapper

def draw(t, img, pose, tracker, observations, ax):
    for axi in ax:
        axi.clear()
        remove_ticks(axi)

    img_fastsam = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_fastsam = np.concatenate([img_fastsam[...,None]]*3, axis=2)

    for i, segment in enumerate(tracker.segments + tracker.segment_graveyard):
        try:
            reconstruction = segment.reconstruction3D(width_height=True)
        except:
            continue
        centroid_w, width, height = reconstruction[:3], reconstruction[3], reconstruction[4]
        centroid_c = transform(np.linalg.inv(pose), centroid_w)
        if centroid_c[2] < 0: # behind camera
            continue
        if i < len(tracker.segments):
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        img = draw_cylinder(img, tracker.camera_params.K, centroid_c, width, height, color=color, id=segment.id)

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
            

    ax[0].imshow(img[...,::-1])
    ax[1].imshow(img_fastsam)
    return


def update(t, img_data, depth_data, pose_data, fastsam, tracker, ax, poses_history):

    try:
        img = img_data.img(t)
        img_depth = depth_data.img(t)
        img_t = img_data.nearest_time(t)
        pose = pose_data.T_WB(img_t)
    except:
        return
      

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
        draw(t, img, pose, tracker, observations, ax)

    poses_history.append(pose)

    return
    

def main(args):
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)

    assert params['bag']['path'] is not None, "bag must be specified in params"
    assert params['bag']['img_topic'] is not None, "img_topic must be specified in params"
    assert params['bag']['cam_info_topic'] is not None, "cam_info_topic must be specified in params"
    assert params['bag']['pose_topic'] is not None, "pose_topic must be specified in params"
    assert params['bag']['pose_time_tol'] is not None, "pose_time_tol must be specified in params"
    if 'depth_compressed_rvl' not in params['bag']:
        params['bag']['depth_compressed_rvl'] = False

    assert params['fastsam']['weights'] is not None, "weights must be specified in params"
    assert params['fastsam']['imgsz'] is not None, "imgsz must be specified in params"
    if 'device' not in params['fastsam']:
        params['fastsam']['device'] = 'cuda'
    
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
    if 'relative_time' in params['bag'] and not params['bag']['relative_time']\
        and 't0' in params['bag'] and 'tf' in params['bag']:
        time_range = [params['bag']['t0'], params['bag']['tf']]
    else:
        time_range = None
    img_data = ImgData(
        data_file=os.path.expanduser(os.path.expandvars(params["bag"]["path"])),
        file_type='bag',
        topic=params["bag"]["img_topic"],
        time_tol=.02,
        time_range=time_range
    )
    img_data.extract_params(params['bag']['cam_info_topic'])

    if 't0' in params['bag'] and params['bag']['relative_time']:
        t0 = img_data.t0 + params['bag']['t0']
    elif 't0' in params['bag']:
        t0 = params['bag']['t0']
    else:
        t0 = img_data.t0

    if 'tf' in params['bag'] and params['bag']['relative_time']:
        tf = img_data.t0 + params['bag']['tf']
    elif 'tf' in params['bag']:
        tf = params['bag']['tf']
    else:
        tf = img_data.tf

    print("Loading depth data for time range {}...".format(time_range))
    depth_data = ImgData(
        data_file=os.path.expanduser(os.path.expandvars(params["bag"]["path"])),
        file_type='bag',
        topic=params['bag']['depth_img_topic'],
        time_tol=.02,
        time_range=time_range,
        compressed=params['bag']['depth_compressed'],
        compressed_rvl=params['bag']['depth_compressed_rvl'],
    )
    depth_data.extract_params(params['bag']['depth_cam_info_topic'])

    print("Loading pose data...")
    T_postmultiply = np.eye(4)
    # if 'T_body_odom' in params['bag']:
    #     T_postmultiply = np.linalg.inv(np.array(params['bag']['T_body_odom']).reshape((4, 4)))
    if 'T_body_cam' in params['bag']:
        T_postmultiply = T_postmultiply @ np.array(params['bag']['T_body_cam']).reshape((4, 4))
    pose_data = PoseData(
        data_file=os.path.expanduser(os.path.expandvars(params['bag']['pose_path'] 
                                    if 'pose_path' in params["bag"] else params["bag"]["path"])),
        file_type='bag',
        topic=params["bag"]["pose_topic"],
        time_tol=params["bag"]["pose_time_tol"],
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
        voxel_size=0.05
    )
    img_area = img_data.camera_params.width * img_data.camera_params.height
    fastsam.setup_filtering(
        ignore_people=True,
        yolo_det_img_size=(128, 128),
        allow_tblr_edges=[True, True, True, True],
        area_bounds=[img_area / 20**2, img_area / 5**2]
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