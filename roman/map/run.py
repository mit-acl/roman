import numpy as np
import cv2 as cv

from os.path import expandvars, expanduser
from typing import Tuple
from dataclasses import dataclass
import time
from copy import deepcopy

from robotdatapy.data.img_data import ImgData
from robotdatapy.data.pose_data import PoseData
from robotdatapy.transform import transform, T_RDFFLU, T_FLURDF
from robotdatapy.data.robot_data import NoDataNearTimeException
from robotdatapy.camera import CameraParams

from roman.object.segment import Segment
from roman.viz import visualize_map_on_img, visualize_observations_on_img
from roman.map.mapper import Mapper, MapperParams
from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.map.map import ROMANMap

@dataclass
class ProcessingTimes:
    fastsam_times: list
    map_times: list
    total_times: list
    
def expandvars_recursive(path):
    """Recursively expands environment variables in the given path."""
    while True:
        expanded_path = expandvars(path)
        if expanded_path == path:
            return expanded_path
        path = expanded_path

class ROMANMapRunner:
    def __init__(self, params: dict, verbose=False, viz_map=False, 
                 viz_observations=False, save_viz=False):
        self.params = self._check_params(params)

        if verbose: print("Extracting time range...")
        self.time_range = self._extract_time_range(params)

        if verbose: 
            print("Loading image data...")
            print(f"Time range: {self.time_range}")
        self.img_data = self._load_img_data()
        if verbose:
            self.t0 = self.img_data.t0
            self.tf = self.img_data.tf

        if verbose: print("Loading depth data for time range {}...".format(self.time_range))
        self.depth_data = self._load_depth_data()

        if verbose: print("Loading pose data...")
        self.pose_data = self._load_pose_data()
        
        if verbose: print("Setting up FastSAM...")
        self.fastsam = self._load_fastsam_wrapper()

        if verbose: print("Setting up ROMAN mapper...")
        self.mapper = self._load_mapper()

        self.verbose = verbose
        self.viz_map = viz_map
        self.viz_observations = viz_observations
        self.save_viz = save_viz
        self.viz_imgs = []
        self.processing_times = ProcessingTimes([], [], [])

    def times(self):
        return np.arange(self.t0, self.tf, self.params['segment_tracking']['dt'])

    def update(self, t: float): 
        t0 = self.img_data.t0
        tf = self.img_data.tf

        if self.verbose: print(f"t: {t - t0:.2f} = {t}")
        img_output = None
        update_t0 = time.time()

        img_time, observations, pose, img = self.update_fastsam(t)
        update_t1 = time.time()
        if observations is not None and pose is not None and img is not None:
            img_output = self.update_segment_track(img_time, observations, pose, img)
        
        update_t2 = time.time()
        self.processing_times.map_times.append(update_t2 - update_t1)
        self.processing_times.fastsam_times.append(update_t1 - update_t0)
        self.processing_times.total_times.append(update_t2 - update_t0)

        return img_output

    def update_fastsam(self, t):

        try:
            img_t = self.img_data.nearest_time(t)
            img = self.img_data.img(img_t)
            img_depth = self.depth_data.img(img_t)
            pose = self.pose_data.T_WB(img_t)
        except NoDataNearTimeException:
            return None, None, None, None
        
        observations = self.fastsam.run(img_t, pose, img, img_depth=img_depth)
        return img_t, observations, pose, img

    def update_segment_track(self, t, observations, pose, img): 

        # collect reprojected masks
        reprojected_bboxs = []
        img_ret = None
        if self.viz_observations:
            segment: Segment
            for i, segment in enumerate(self.mapper.segments + self.mapper.inactive_segments):
                bbox = segment.reprojected_bbox(pose)
                if bbox is not None:
                    reprojected_bboxs.append((segment.id, bbox))

        if len(observations) > 0:
            self.mapper.update(t, pose, observations)
        else:
            self.mapper.poses_flu_history.append(pose @ self.mapper._T_camera_flu)
            self.mapper.times_history.append(t)

        if self.viz_map or self.viz_observations:
            img_ret = self.draw(t, img, pose,observations, reprojected_bboxs)

        if self.save_viz:
            self.viz_imgs.append(img_ret)

        return img_ret

    def _check_params(self, params: dict) -> dict:
        """
        Checks that required parameters are set and sets default values for optional parameters.
        TODO: This should just become a param class with defaults. It's pretty messy right now.

        Args:
            params (dict): param dictionary

        Returns:
            dict: checked and updated param dictionary (with defaults)
        """

        # TODO: this is getting really messy. Need a way to clean this up.
        if 'time' in params:
            assert 'relative' in params['time'], "relative must be specified in params"
            assert 't0' in params['time'], "t0 must be specified in params"
            assert 'tf' in params['time'], "tf must be specified in params"

        # use_kitti = params['use_kitti']
        try:
            use_kitti = params['use_kitti']
        except KeyError:
            params['use_kitti'] = False
            use_kitti = False

        if not use_kitti:
            assert params['img_data']['path'] is not None, "bag must be specified in params"
            assert params['img_data']['img_topic'] is not None, "img_topic must be specified in params"
            assert params['img_data']['cam_info_topic'] is not None, "cam_info_topic must be specified in params"
            if 'color_compressed' not in params['img_data']:
                params['img_data']['color_compressed'] = True
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
            params['fastsam']['max_depth'] = 7.5
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
        if 'rotate_img' not in params['fastsam']:
            params['fastsam']['rotate_img'] = None
        if 'clip' not in params['fastsam']:
            params['fastsam']['clip'] = True
        if 'yolo' not in params:
            params['yolo'] = {'imgsz': params['fastsam']['imgsz']}
            
        # IMG_DATA PARAMS TODO: cleanup
        if 'depth_scale' not in params['img_data']:
            params['img_data']['depth_scale'] = 1e3
        
        
        assert params['segment_tracking']['min_iou'] is not None, "min_iou must be specified in params"
        assert params['segment_tracking']['min_sightings'] is not None, "min_sightings must be specified in params"
        assert params['segment_tracking']['max_t_no_sightings'] is not None, "max_t_no_sightings must be specified in params"
        assert params['segment_tracking']['mask_downsample_factor'] is not None, "Mask downsample factor cannot be none!"

        return params

    def _extract_time_range(self, params: dict) -> Tuple[float, float]:
        """
        Uses the params dictionary and image data to set an absolute time range for the data.

        Args:
            params (dict): Params dict.

        Returns:
            Tuple[float, float]: Beginning and ending time (or none).
        """
        if params['use_kitti']:
            time_range = [params['time']['t0'], params['time']['tf']]
        else:
            img_file_path = expanduser(expandvars_recursive(params["img_data"]["path"]))
            if 'time' in params:
                if 'relative' in params['time'] and params['time']['relative']:
                    topic_t0 = ImgData.topic_t0(img_file_path, expandvars_recursive(params["img_data"]["img_topic"]))
                    time_range = [topic_t0 + params['time']['t0'], topic_t0 + params['time']['tf']]
                else:
                    time_range = [params['time']['t0'], params['time']['tf']]
            else:
                time_range = None
        return time_range

    def _load_img_data(self) -> ImgData:
        """
        Loads image data.

        Returns:
            ImgData: Image data object.
        """
        if self.params['use_kitti']:
            img_data = ImgData.from_kitti(self.params['img_data']['base_path'], 'rgb')
            img_data.extract_params()
        else:
            img_file_path = expanduser(expandvars_recursive(self.params["img_data"]["path"]))
            img_data = ImgData.from_bag(
                path=img_file_path,
                topic=expandvars_recursive(self.params["img_data"]["img_topic"]),
                time_tol=self.params['segment_tracking']['dt'] / 2.0,
                time_range=self.time_range,
                compressed=self.params['img_data']['color_compressed'],
                compressed_encoding='bgr8'
            )
            img_data.extract_params(expandvars_recursive(self.params['img_data']['cam_info_topic']))
        return img_data

    def _load_depth_data(self) -> ImgData:
        """
        Loads depth image data.

        Returns:
            ImgData: Image data object containing depth images.
        """
        if self.params['use_kitti']:
            depth_data = ImgData.from_kitti(self.params['img_data']['base_path'], 'depth')
            depth_data.extract_params()
        else:
            img_file_path = expanduser(expandvars_recursive(self.params["img_data"]["path"]))
            depth_data = ImgData.from_bag(
                path=img_file_path,
                topic=expandvars_recursive(self.params['img_data']['depth_img_topic']),
                time_tol=.05,
                time_range=self.time_range,
                compressed=self.params['img_data']['depth_compressed'],
                compressed_rvl=self.params['img_data']['depth_compressed_rvl'],
            )
            depth_data.extract_params(expandvars_recursive(self.params['img_data']['depth_cam_info_topic']))
        
        return depth_data
    
    def _load_pose_data(self) -> PoseData:
        """
        Loads pose data.

        Returns:
            PoseData: Pose data object.
        """
        if not self.params['use_kitti']:
            pose_file_path = expandvars_recursive(self.params['pose_data']['path'])
            pose_file_type = self.params['pose_data']['file_type']
            pose_time_tol = self.params['pose_data']['time_tol']
            if pose_file_type == 'bag':
                pose_topic = expandvars_recursive(self.params['pose_data']['topic'])
                csv_options = None
            elif pose_file_type == 'csv':
                pose_topic = None
                csv_options = self.params['pose_data']['csv_options']
            
        if 'T_odombase_camera' in self.params['pose_data']:
            T_postmultiply = self._find_transformation(self.params['pose_data']['T_odombase_camera'])
        else:
            T_postmultiply = None
        
        if self.params['use_kitti']:
            pose_data = PoseData.from_kitti(self.params['pose_data']['base_path'])
        elif pose_file_type == 'bag':
            pose_data = PoseData.from_bag(
                path=expanduser(pose_file_path),
                topic=expandvars_recursive(pose_topic),
                time_tol=pose_time_tol,
                interp=True,
                T_postmultiply=T_postmultiply
            )
        elif pose_file_type == 'bag_tf':
            pose_data = PoseData.from_bag_tf(
                path=expanduser(pose_file_path),
                parent_frame=expandvars_recursive(self.params['pose_data']['parent']),
                child_frame=expandvars_recursive(self.params['pose_data']['child']),
                time_tol=pose_time_tol,
                interp=True,
                T_postmultiply=T_postmultiply
            )
        else:
            pose_data = PoseData.from_csv(
                path=expanduser(pose_file_path),
                csv_options=csv_options,
                time_tol=pose_time_tol,
                interp=True,
                T_postmultiply=T_postmultiply
            )
        
        return pose_data
    
    def _find_transformation(self, param_dict) -> np.array:
        """
        Finds the transformation from the body frame to the camera frame.

        Returns:
            np.array: Transformation matrix.
        """
        # T_postmultiply = np.eye(4)
        # # if 'T_body_odom' in param_dict:
        # #     T_postmultiply = np.linalg.inv(np.array(param_dict['T_body_odom']).reshape((4, 4)))
        # if 'T_body_cam' in param_dict:
        #     T_postmultiply = T_postmultiply @ np.array(param_dict['T_body_cam']).reshape((4, 4))
        if param_dict['input_type'] == 'string':
            if param_dict['string'] == 'T_FLURDF':
                return T_FLURDF
            elif param_dict['string'] == 'T_RDFFLU':
                return T_RDFFLU
            else:
                raise ValueError("Invalid string.")
        elif param_dict['input_type'] == 'tf':
            img_file_path = expanduser(expandvars_recursive(self.params["img_data"]["path"]))
            T = PoseData.static_tf_from_bag(
                expanduser(img_file_path), 
                expandvars_recursive(param_dict['parent']), 
                expandvars_recursive(param_dict['child'])
            )
            if param_dict['inv']:
                return np.linalg.inv(T)
            else:
                return T
        else:
            raise ValueError("Invalid input type.")

    def _load_fastsam_wrapper(self) -> FastSAMWrapper:
        """
        Creates a FastSAMWrapper object from the params dictionary.

        Returns:
            FastSAMWrapper: Wrapper for extracting observations from images using FastSAM.
        """
        fastsam = FastSAMWrapper(
            weights=expanduser(expandvars_recursive(self.params['fastsam']['weights'])),
            imgsz=self.params['fastsam']['imgsz'],
            device=self.params['fastsam']['device'],
            mask_downsample_factor=self.params['segment_tracking']['mask_downsample_factor'],
            rotate_img=self.params['fastsam']['rotate_img']
        )
        fastsam.setup_rgbd_params(
            depth_cam_params=self.depth_data.camera_params, 
            max_depth=self.params['fastsam']['max_depth'],
            depth_scale=self.params['img_data']['depth_scale'],
            voxel_size=0.05,
            erosion_size=self.params['fastsam']['erosion_size'],
            plane_filter_params=self.params['fastsam']['plane_filter_params']
        )
        img_area = self.img_data.width * self.img_data.height
        fastsam.setup_filtering(
            ignore_labels=self.params['fastsam']['ignore_labels'],
            use_keep_labels=self.params['fastsam']['use_keep_labels'],
            keep_labels=self.params['fastsam']['keep_labels'],
            keep_labels_option=self.params['fastsam']['keep_labels_option'],
            yolo_det_img_size=self.params['yolo']['imgsz'],
            allow_tblr_edges=[True, True, True, True],
            area_bounds=[img_area / (self.params['fastsam']['min_mask_len_div']**2), img_area / (self.params['fastsam']['max_mask_len_div']**2)],
            clip_embedding=self.params['fastsam']['clip']
        )

        return fastsam

    def _load_mapper(self) -> Mapper:
        """
        Creates a Mapper object from the params dictionary.

        Returns:
            Mapper: ROMAN mapper.
        """
        params = MapperParams(
            camera_params=self.img_data.camera_params,
            min_iou=self.params['segment_tracking']['min_iou'],
            min_sightings=self.params['segment_tracking']['min_sightings'],
            max_t_no_sightings=self.params['segment_tracking']['max_t_no_sightings'],
            mask_downsample_factor=self.params['segment_tracking']['mask_downsample_factor']
        )
        mapper = Mapper(params)
        if 'T_camera_flu' in self.params['pose_data']:
            mapper.set_T_camera_flu(self._find_transformation(self.params['pose_data']['T_camera_flu']))
        else:
            mapper.set_T_camera_flu(np.eye(4))
        return mapper

    def draw(self, t, img, pose, observations, reprojected_bboxs):
        img_orig = deepcopy(img)
        
        if len(img_orig.shape) == 2:
            img = np.concatenate([img_orig[...,None]]*3, axis=2)
            
        if self.viz_map:
            img = visualize_map_on_img(t, pose, img, self.mapper)

        if self.viz_observations:
            img_fastsam = visualize_observations_on_img(t, img_orig, self.mapper, observations, reprojected_bboxs)
                
        # rotate images
        img = self.fastsam.apply_rotation(img)
        if self.viz_observations:
            img_fastsam = self.fastsam.apply_rotation(img_fastsam)
                
        # concatenate images
        if self.viz_observations and self.viz_map:
            img_ret = np.concatenate([img, img_fastsam], axis=1)
        elif self.viz_map:
            img_ret = img
        elif self.viz_observations:
            img_ret = img_fastsam
        return img_ret