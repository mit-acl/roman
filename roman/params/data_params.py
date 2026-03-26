###########################################################
#
# data_params.py
#
# Parameter class for data loading
#
# Authors: Mason Peterson, Qingyuan Li
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
from dataclasses import dataclass
import yaml
from typing import List, Tuple
from functools import cached_property

from robotdatapy.data import ImgData, PoseData, PointCloudData
from robotdatapy.transform import T_FLURDF, T_RDFFLU

from roman.utils import expandvars_recursive, combinedicts_recursive

def find_transformation(bag_path, param_dict) -> np.array:
    """
    Converts a transform parameter dictionary into a transformation matrix.

    Returns:
        np.array: Transformation matrix.
    """
    if param_dict['input_type'] == 'string':
        if param_dict['string'] == 'T_FLURDF':
            return T_FLURDF
        elif param_dict['string'] == 'T_RDFFLU':
            return T_RDFFLU
        else:
            raise ValueError("Invalid string.")
    elif param_dict['input_type'] == 'tf':
        bag_path = expandvars_recursive(bag_path)
        # by default looks for a static tf, but if the user wants to reference a tf that is
        # theoretically static, but is published under /tf, then 'include_non_static_tf' can be set.
        if param_dict.get('include_non_static_tf', False):
            tf_data = PoseData.from_bag_tf(
                expandvars_recursive(bag_path),
                expandvars_recursive(param_dict['parent']),
                expandvars_recursive(param_dict['child'])
            )
            T = tf_data.pose(tf_data.t0)
        else:
            T = PoseData.any_static_tf_from_bag(
                expandvars_recursive(bag_path),
                expandvars_recursive(param_dict['parent']),
                expandvars_recursive(param_dict['child'])
            )
        if 'inv' in param_dict.keys() and param_dict['inv']:
            T = np.linalg.inv(T)
        return T
    elif param_dict['input_type'] == 'matrix':
        return np.array(param_dict['matrix']).reshape((4, 4))
    else:
        raise ValueError("Invalid input type.")    

@dataclass
class PointCloudDataParams:
    path: str
    topic: str
    T_camera_rangesense: None
    ignore_ros_time: bool = False

    @classmethod
    def from_dict(cls, params_dict: dict):
        if 'T_camera_rangesense' in params_dict:
            params_dict['T_camera_rangesense'] = find_transformation(bag_path=params_dict['path'],
                                                                     param_dict=params_dict['T_camera_rangesense'])
        else:
            params_dict['T_camera_rangesense'] = None
        return cls(**params_dict)

@dataclass
class PoseDataParams:
    
    params_dict: dict
    T_camera_flu_dict: dict
    T_odombase_camera_dict: dict = None
    
    @classmethod
    def from_dict(cls, params_dict: dict):
        params_dict_subset = {k: v for k, v in params_dict.items() 
                       if k != 'T_camera_flu' and k != 'T_odombase_camera'}
        T_camera_flu_dict = params_dict['T_camera_flu']
        T_odombase_camera_dict = params_dict['T_odombase_camera'] \
            if 'T_odombase_camera' in params_dict else None
        return cls(params_dict=params_dict_subset, T_camera_flu_dict=T_camera_flu_dict, 
                   T_odombase_camera_dict=T_odombase_camera_dict)
        
    @property
    def T_camera_flu(self) -> np.array:
        return self._find_transformation(self.T_camera_flu_dict)
    
    @property
    def T_odombase_camera(self) -> np.array:
        if self.T_odombase_camera_dict is not None:
            return self._find_transformation(self.T_odombase_camera_dict)
        else:
            return np.eye(4)
        
    @property
    def odombase_frame(self) -> str:
        return self.T_odombase_camera_dict['parent'] if not self.T_odombase_camera_dict['inv'] else self.T_odombase_camera_dict['child']
        
    def load_pose_data(self, extra_key_vals: dict) -> PoseData:
        """
        Loads pose data.

        Returns:
            PoseData: Pose data object.
        """
        params_dict = {k: v for k, v in self.params_dict.items()}
        for k, v in extra_key_vals.items():
            params_dict[k] = v
            
        # expand variables
        for k, v in params_dict.items():
            if type(params_dict[k]) == str:
                params_dict[k] = expandvars_recursive(params_dict[k])
        pose_data = PoseData.from_dict(params_dict)
        return pose_data
    
    def _find_transformation(self, tf_dict) -> np.array:
        return find_transformation(self.params_dict["path"], tf_dict)
    
@dataclass
class DataParams:
    
    img_data_params: dict
    depth_data_params: dict
    pointcloud_data_params: PointCloudDataParams
    pose_data_params: PoseDataParams
    use_pointcloud: bool = False
    dt: float = 1/6
    runs: list = None
    run_env: str = None
    time_params: dict = None
    max_time: float = None
    kitti: bool = False
    
    def __post_init__(self):
        if self.time_params is not None:
            assert 'relative' in self.time_params, "relative must be specified in params"
            assert 't0' in self.time_params, "t0 must be specified in params"
            assert 'tf' in self.time_params, "tf must be specified in params"
        
        # set default img data type for backwards compatibility
        for params in [self.img_data_params, self.depth_data_params]:
            # continue if depth data params is None
            if not params:
                continue
            params['time_tol'] = self.dt / 2.0
            if 'type' not in params:
                params['type'] = 'bag'

    @classmethod
    def from_yaml(cls, yaml_path: str, run: str = None):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if run is None:
            return cls(
                None, None, None, None,
                use_pointcloud=data['depth_source'] == 'pointcloud' if 'depth_source' in data else \
                            (data['use_pointcloud'] if 'use_pointcloud' in data else False),
                dt=data['dt'] if 'dt' in data else 1/6,
                runs=data['runs'] if 'runs' in data else None,
                run_env=data['run_env'] if 'run_env' in data else None,
                max_time=data['max_time'] if 'max_time' in data else None
            )
        elif run in data:
            run_data = combinedicts_recursive(data, data[run])
        else:
            run_data = data
        return cls(
            run_data['img_data'],
            run_data['depth_data'] if 'depth_data' in run_data else None,
            PointCloudDataParams.from_dict(run_data['pointcloud_data']) if 'pointcloud_data' in run_data else None,
            PoseDataParams.from_dict(run_data['pose_data']),
            use_pointcloud=run_data['depth_source'] == 'pointcloud' if 'depth_source' in run_data else \
                          (run_data['use_pointcloud'] if 'use_pointcloud' in run_data else False),
            dt=run_data['dt'] if 'dt' in run_data else 1/6,
            runs=data['runs'] if 'runs' in data else None,
            run_env=data['run_env'] if 'run_env' in data else None,
            time_params=run_data['time'] if 'time' in run_data else None,
            max_time=run_data['max_time'] if 'max_time' in run_data else None,
            kitti=run_data['kitti'] if 'kitti' in run_data else False
        )
        
    @property
    def time_range(self) -> Tuple[float, float]:
        return self._extract_time_range()
    
    def load_pose_data(self) -> PoseData:
        """
        Loads pose data.

        Returns:
            PoseData: Pose data object.
        """        
        if self.pose_data_params.T_odombase_camera is not None:
            T_postmultiply = self.pose_data_params.T_odombase_camera
        else:
            T_postmultiply = None
            
        extra_key_vals={'T_postmultiply': T_postmultiply, 'interp': True}
            
        return self.pose_data_params.load_pose_data(extra_key_vals)
        
    def load_pointcloud_data(self) -> PointCloudData:
        """
        Loads point cloud data.

        Returns:
            PointCloudData: PointCloud 
        """
        return PointCloudData.from_bag(
            path=expandvars_recursive(self.pointcloud_data_params.path),
            topic=expandvars_recursive(self.pointcloud_data_params.topic),
            time_tol=self.dt / 2.0,
            time_range=self.time_range,
            ignore_ros_time=self.pointcloud_data_params.ignore_ros_time
        )

    def load_img_data(self) -> ImgData:
        """
        Loads image data.
        
        Args:
            time_range (List[float, float]): Time range to load image data.

        Returns:
            ImgData: Image data object.
        """
        return self._load_img_data(color=True)
    
    def load_depth_data(self) -> ImgData:
        """
        Loads depth data.
        
        Args:
            time_range (List[float, float]): Time range to load depth data.

        Returns:
            ImgData: Depth data object.
        """
        return self._load_img_data(color=False)
        
    def _load_img_data(self, color=True) -> ImgData:
        """
        Loads color or depth image data.

        Args:
            color (bool, optional): True if color, False if depth. Defaults to True.

        Returns:
            ImgData: Image data object.
        """
        if color:
            img_data_params = self.img_data_params
        else:
            img_data_params = self.depth_data_params

        # TODO: This has to be here right now because other files 
        # externally change time_params - this should probably have a
        # function for externally changing time_params instead
        img_data_params['time_range'] = self.time_range

        # expand env variables
        for k, v in img_data_params.items():
            if isinstance(v, str):
                img_data_params[k] = expandvars_recursive(v)

        img_data = ImgData.from_dict(img_data_params)
        
        if img_data_params['type'] == 'kitti':
            img_data.extract_params()
        return img_data
    
    def _extract_time_range(self) -> Tuple[float, float]:
        """
        Uses the params dictionary and image data to set an absolute time range for the data.

        Args:
            params (dict): Params dict.

        Returns:
            Tuple[float, float]: Beginning and ending time (or none).
        """
        if self.kitti:
            time_range = [self.time_params['t0'], self.time_params['tf']]
        else:
            if self.time_params is not None:
                if self.time_params['relative']:
                    topic_t0 = self.data_t0
                    time_range = [topic_t0 + self.time_params['t0'], 
                                  topic_t0 + self.time_params['tf']]
                else:
                    time_range = [self.time_params['t0'], 
                                  self.time_params['tf']]
            else:
                time_range = None
        return time_range
    
    @cached_property
    def data_t0(self) -> float:
        return self.data_t_range[0]
    
    @cached_property
    def data_tf(self) -> float:
        return self.data_t_range[1]
    
    @cached_property
    def data_t_range(self) -> float:
        if self.img_data_params['type'] in ['bag', 'bag2']:
            return ImgData.topic_t_range(self.img_data_params['path'], 
                                     self.img_data_params['topic'])
        else:
            raise NotImplementedError("data_t_range not implemented for this img data type")
        