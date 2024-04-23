# python3 save_observations.py -b /home/masonbp/data/west_point_2023/1117_04_loc_tower_loop_far_side_loop_o3_CCW/align.bag --pose-topic "/acl_jackal2/kimera_vio_ros/odometry" --camera-topic "/acl_jackal2/forward/color" --t0 315 --tf 320 -o /home/masonbp/results/west_point_2023/segment_observations/test00.pkl

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import tqdm

from robot_utils.robot_data.img_data import ImgData
from robot_utils.robot_data.pose_data import PoseData
from robot_utils.transform import transform
from robot_utils.robot_data.general_data import GeneralData

from segment_track.observation import Observation
from segment_track.segment import Segment
from segment_track.tracker import Tracker
from segment_track.fastsam_wrapper import FastSAMWrapper


def extract_observations(t0, tf, img_data, pose_data, fastsam):
    ts = []
    all_observations = []
    for t in tqdm.tqdm(np.arange(t0, tf, .05)):
        try:
            img = img_data.img(t)
            img_t = img_data.nearest_time(t)
            pose = pose_data.T_WB(img_t)
        except:
            continue
        observations = fastsam.run(t, pose, img)
        all_observations.append(observations)
        ts.append(t)
    observations = GeneralData(data_type='list', data=all_observations, times=ts)
    return observations

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--bag', type=str, help='Path to bag file', required=True)
    parser.add_argument('--camera-topic', type=str, help='Camera topic', 
                        default="/RR08/d455/color/image_raw/compressed")
    parser.add_argument('--pose-topic', type=str, help='Pose topic',
                        default="/RR08/world")
    parser.add_argument('--pose-time-tol', type=float, help='Pose time tolerance', default=.4)
    parser.add_argument('--t0', default=None, type=float, help='Start time')
    parser.add_argument('--tf', default=None, type=float, help='End time')
    parser.add_argument('-o', '--output', type=str, help='Path to output file', required=True)
    args = parser.parse_args()    

    if "image_raw" not in args.camera_topic:
        args.camera_info_topic = args.camera_topic + "/camera_info"
        args.camera_topic = args.camera_topic + "/image_raw/compressed"
    else:
        args.camera_info_topic = args.camera_topic.split("image_raw")[0] + "camera_info"

    img_data = ImgData(
        data_file=args.bag,
        file_type='bag',
        topic=args.camera_topic,
        time_tol=.02,
    )
    img_data.extract_params(args.camera_info_topic)
    # cam_params_file = "/home/masonbp/data/west_point_2023/acl_jackal2_cam_params.pkl"
    # pkl_file = open(cam_params_file, 'wb')
    # pickle.dump(img_data.camera_params, pkl_file)
    # pkl_file.close()    

    pose_data = PoseData(
        data_file=args.bag,
        file_type='bag',
        topic=args.pose_topic,
        time_tol=args.pose_time_tol,
        interp=True
    )

    fastsam = FastSAMWrapper(
        weights="/home/lucas/Documents/GitLab/dmot/fastsam3d/FastSAM/weights/FastSAM-x.pt",
        imgsz=512,
        device='cuda'
    )
    img_area = img_data.camera_params.width * img_data.camera_params.height
    fastsam.setup_filtering(
        ignore_people=True,
        yolo_det_img_size=(512, 512),
        allow_tblr_edges=[True, True, True, True],
        area_bounds=[img_area / 20**2, img_area / 3**2]
    )

    if args.t0 is None:
        args.t0 = img_data.t0
    else:
        args.t0 = img_data.t0 + args.t0

    if args.tf is None:
        args.tf = img_data.tf
    else:
        args.tf = img_data.t0 + args.tf

    observations = extract_observations(args.t0, args.tf, img_data, pose_data, fastsam)

    pkl_file = open(args.output, 'wb')
    pickle.dump(observations, pkl_file)
    pkl_file.close()