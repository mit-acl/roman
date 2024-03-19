import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import pickle
import tqdm
import yaml
import time
import cv2 as cv

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

def draw(t, img, pose, tracker, ax):
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

    for segment in tracker.segments:
        if segment.last_seen == t:
            colored_mask = np.zeros_like(img)
            rand_color = np.random.randint(0, 255, 3)
            # print(rand_color)
            colored_mask = segment.last_mask.astype(np.int32)[..., np.newaxis]*rand_color
            colored_mask = colored_mask.astype(np.uint8)
            img_fastsam = cv.addWeighted(img_fastsam, 1.0, colored_mask, 0.5, 0)
            for obs in segment.observations[::-1]:
                if obs.time == t:
                    img_fastsam = cv.putText(img_fastsam, str(segment.id), obs.pixel.astype(np.int32), 
                         cv.FONT_HERSHEY_SIMPLEX, 0.5, rand_color.tolist(), 2)
                    break
            

    ax[0].imshow(img[...,::-1])
    ax[1].imshow(img_fastsam)
    return


def update(t, img_data, pose_data, fastsam, tracker, ax):

    try:
        img = img_data.img(t)
        img_t = img_data.nearest_time(t)
        pose = pose_data.T_WB(img_t)
    except:
        return
    observations = fastsam.run(t, pose, img)

    if np.round(t, 1) % 1 < 1e-3:
        tracker.merge()

    print(f"observations: {len(observations)}")

    if len(observations) > 0:
        tracker.update(t, pose, observations)

    print(f"segments: {len(tracker.segments)}")

    draw(t, img, pose, tracker, ax)

    return

    

def main(args):
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)

    assert params['bag']['path'] is not None, "bag must be specified in params"
    assert params['bag']['img_topic'] is not None, "img_topic must be specified in params"
    assert params['bag']['cam_info_topic'] is not None, "cam_info_topic must be specified in params"
    assert params['bag']['pose_topic'] is not None, "pose_topic must be specified in params"
    assert params['bag']['pose_time_tol'] is not None, "pose_time_tol must be specified in params"

    assert params['fastsam']['weights'] is not None, "weights must be specified in params"
    assert params['fastsam']['imgsz'] is not None, "imgsz must be specified in params"
    
    assert params['segment_tracking']['min_iou'] is not None, "min_iou must be specified in params"
    assert params['segment_tracking']['min_sightings'] is not None, "min_sightings must be specified in params"
    assert params['segment_tracking']['pixel_std_dev'] is not None, "max_t_no_sightings must be specified in params"
    assert params['segment_tracking']['max_t_no_sightings'] is not None, "max_t_no_sightings must be specified in params"

    print("Loading image data...")
    if 'relative_time' in params['bag'] and not params['bag']['relative_time']\
        and 't0' in params['bag'] and 'tf' in params['bag']:
        time_range = [params['bag']['t0'], params['bag']['tf']]
    else:
        time_range = None
    img_data = ImgData(
        data_file=params["bag"]["path"],
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

    print("Loading pose data...")
    T_postmultiply = np.eye(4)
    # if 'T_body_odom' in params['bag']:
    #     T_postmultiply = np.linalg.inv(np.array(params['bag']['T_body_odom']).reshape((4, 4)))
    if 'T_body_cam' in params['bag']:
        T_postmultiply = T_postmultiply @ np.array(params['bag']['T_body_cam']).reshape((4, 4))
    pose_data = PoseData(
        data_file=params['bag']['pose_path'] if 'pose_path' in params["bag"] else params["bag"]["path"],
        file_type='bag',
        topic=params["bag"]["pose_topic"],
        time_tol=params["bag"]["pose_time_tol"],
        interp=True,
        T_postmultiply=T_postmultiply
    )
    

    print("Setting up FastSAM...")
    fastsam = FastSAMWrapper(
        weights=params['fastsam']['weights'],
        imgsz=params['fastsam']['imgsz'],
        device='cuda'
    )
    img_area = img_data.camera_params.width * img_data.camera_params.height
    fastsam.setup_filtering(
        ignore_people=True,
        yolo_det_img_size=(128, 128),
        allow_tblr_edges=[False, False, False, False],
        area_bounds=[img_area / 20**2, img_area / 5**2]
    )

    print("Setting up segment tracker...")
    tracker = Tracker(
        camera_params=img_data.camera_params,
        pixel_std_dev=params['segment_tracking']['pixel_std_dev'],
        min_iou=params['segment_tracking']['min_iou'],
        min_sightings=params['segment_tracking']['min_sightings'],
        max_t_no_sightings=params['segment_tracking']['max_t_no_sightings']
    )

    print("Running segment tracking!")
    wc_t0 = time.time()
    fig, ax = plt.subplots(1, 2, dpi=400)
    def update_wrapper(t): update(t, img_data, pose_data, fastsam, tracker, ax); print(f"t: {t - t0:.2f}")

    if not args.no_vid:
        ani = FuncAnimation(fig, update_wrapper, frames=tqdm.tqdm(np.arange(t0, tf, params['segment_tracking']['dt'])), interval=10, repeat=False)
        if not args.output:
            plt.show()
        else:
            video_file = args.output + ".mp4"
            writervideo = FFMpegWriter(fps=30)
            ani.save(video_file, writer=writervideo)

            pkl_path = args.output + ".pkl"
            pkl_file = open(pkl_path, 'wb')
            pickle.dump(tracker, pkl_file, -1)
            pkl_file.close()
    else:
        for t in np.arange(t0, tf, params['segment_tracking']['dt']):
            update_wrapper(t)

    print(f"Segment tracking took {time.time() - wc_t0:.2f} seconds")
    print(f"Run duration was {tf - t0:.2f} seconds")
    print(f"Compute per second: {(time.time() - wc_t0) / (tf - t0):.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, help='Path to params file', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to output file', required=False, default=None)
    parser.add_argument('--no-vid', action='store_true', help='Do not show or save video')
    args = parser.parse_args()

    main(args)