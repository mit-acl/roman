import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import math
import argparse
from dataclasses import dataclass
from typing import Union, List, Tuple, Dict
import robotdatapy as rdp
from robotdatapy.camera import CameraParams
import shapely

from roman.align.results import SubmapAlignResults
from roman.map.map import ROMANMap
from roman.utils import expandvars_recursive

ALIGN_RESULTS_MATRIX_ATTRIBUTES = [
    'robots_nearby_mat',
    'clipper_angle_mat',
    'clipper_dist_mat',
    'clipper_num_associations',
    'similarity_mat',
    'submap_yaw_diff_mat',
    'T_ij_mat',
    'T_ij_hat_mat',
]

STANDARD_YAW_DIFFS = {
    "all": (0.0, 180.0),
    "same_direction": (0.0, 60.0),
    "perpendicular": (60.0, 120.0),
    "opposite_direction": (120.0, 180.0),
}

@dataclass
class EvalParams:
    angular_err_thresh_deg: float = 5.0
    distance_err_thresh_m: float = 1.0
    evaluation_distance_m: float = 10.0
    robot_names: List[str] = None
    include_inter_robot: bool = False
    rm_non_camera_overlap: bool = False
    cam_view_dist_bounds: Tuple[float, float] = (0.0, 20.0)
    T_ij_uses_rdf: bool = False
    
    def __post_init__(self):
        assert not self.include_inter_robot, "Inter-robot evaluation not implemented yet."

    @property
    def robot_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for i in range(len(self.robot_names)):
            start_idx = i if self.include_inter_robot else i + 1
            for j in range(start_idx, len(self.robot_names)):
                pairs.append((self.robot_names[i], self.robot_names[j]))
        return pairs

    @property
    def robot_pairs_as_strings(self) -> List[str]:
        return [f"{r1}_{r2}" for r1, r2 in self.robot_pairs]
    
@dataclass
class EvalInput:
    directory: str
    name: str = None
    map_directory: str = None
    params_directory: str = None

    def get_directory(self):
        assert os.path.isdir(self.directory), f"Directory {self.directory} does not exist."
        if os.path.isdir(f"{self.directory}/align"):
            return f"{self.directory}/align"
        return self.directory

    def get_name(self):
        if self.name is not None:
            return self.name
        return os.path.basename(os.path.normpath(self.directory))
    
    def get_map_dir(self):
        return self._get_dir("map", self.map_directory)
    
    def _get_dir(self, basename: str, first_try: str = None):
        if first_try is not None:
            assert os.path.isdir(first_try), f"Directory {first_try} does not exist."
            return first_try
        dir_option1 = f"{self.directory}/{basename}"
        dir_option2 = f"{self.directory}/../{basename}"
        assert os.path.isdir(dir_option1) or os.path.isdir(dir_option2), \
            f"Directory does not exist in either {dir_option1} or {dir_option2}."
        if os.path.isdir(dir_option1):
            return dir_option1
        return dir_option2
    

class SubmapAlignEvaluator():

    def __init__(self, params: EvalParams):
        self.params = params
        self.results: Dict[str, SubmapAlignResults] = {}
        self.robot_camera_params = {robot: None for robot in self.params.robot_names}

    def load_results(self, eval_inputs: Union[EvalInput, List[EvalInput]]):
        if isinstance(eval_inputs, EvalInput):
            eval_inputs = [eval_inputs]
        for eval_input in eval_inputs:
            result_paths = self._get_results_paths(eval_input)
            combined_results = None
            for path, robots in zip(result_paths, self.params.robot_pairs):
                new_result = SubmapAlignResults.load(path)
                if self.params.rm_non_camera_overlap:
                    new_result = self._rm_non_camera_overlap(new_result, robots, eval_input)
 
                for matrix_attribute in ALIGN_RESULTS_MATRIX_ATTRIBUTES:
                    if hasattr(new_result, matrix_attribute):
                        matrix = getattr(new_result, matrix_attribute).reshape((-1))
                        setattr(new_result, matrix_attribute, matrix)

                if combined_results is None:
                    combined_results = new_result
                else:
                    for matrix_attribute in ALIGN_RESULTS_MATRIX_ATTRIBUTES:
                        if hasattr(new_result, matrix_attribute):
                            combined_matrix = getattr(combined_results, matrix_attribute)
                            new_matrix = getattr(new_result, matrix_attribute)
                            combined_matrix = np.concatenate((combined_matrix, new_matrix), axis=0)
                            setattr(combined_results, matrix_attribute, combined_matrix)

            self.results[eval_input.get_name()] = combined_results

    def evaluate_align_success_rate(
        self,
        yaw_diff_min_deg: float = 0.0,
        yaw_diff_max_deg: float = 180.0,
    ) -> Dict[str, float]:
        success_rates = {}
        for name, results in self.results.items():
            relevant_alignments = (
                (results.robots_nearby_mat <= self.params.evaluation_distance_m) &
                (results.submap_yaw_diff_mat >= yaw_diff_min_deg) &
                (results.submap_yaw_diff_mat <= yaw_diff_max_deg)
            )
            correct_alignments = (
                (results.clipper_angle_mat <= self.params.angular_err_thresh_deg) &
                (results.clipper_dist_mat <= self.params.distance_err_thresh_m)
            )
            num_relevant = np.nansum(relevant_alignments)
            num_correct = np.nansum(relevant_alignments & correct_alignments)
            success_rate = num_correct / num_relevant if num_relevant > 0 else float('nan')
            success_rates[name] = success_rate
        return success_rates


    def _get_results_paths(self, eval_input: EvalInput) -> List[str]:
        dir_path = eval_input.get_directory()
        result_files = []
        for robot_pair in self.params.robot_pairs_as_strings:
            file_path = os.path.join(dir_path, robot_pair, "align.pkl")
            assert os.path.isfile(file_path), f"Result file {file_path} does not exist."
            result_files.append(file_path)
        
        return result_files
    
    def _rm_non_camera_overlap(
            self, 
            results: SubmapAlignResults, 
            robots: Tuple[str, str], 
            eval_input: EvalInput
        ) -> SubmapAlignResults:
        for i in range(results.robots_nearby_mat.shape[0]):
            for j in range(results.robots_nearby_mat.shape[1]):
                if not np.isnan(results.robots_nearby_mat[i,j]) and \
                not self._camera_views_overlap(
                    results.T_ij_mat[i,j],
                    self._get_robot_camera_params(robots[0], eval_input),
                    self._get_robot_camera_params(robots[1], eval_input)
                ):
                    results.robots_nearby_mat[i,j] = np.nan
        return results

    def _get_robot_camera_params(self, robot: str, eval_input: EvalInput) -> CameraParams:
        if self.robot_camera_params[robot] is not None:
            return self.robot_camera_params[robot]
        
        robot_map_path = f"{eval_input.get_map_dir()}/{robot}.pkl"

        # get a random segment to extract camera params
        roman_map = ROMANMap.from_pickle(robot_map_path)
        cam_params = roman_map.segments[0].camera_params
        self.robot_camera_params[robot] = cam_params
        return cam_params

    def _camera_views_overlap(
        self, 
        T_cam1_cam2: np.ndarray, 
        cam1_params: CameraParams, 
        cam2_params: CameraParams
    ) -> bool:

        # T_ij in cam frame
        if self.params.T_ij_uses_rdf:
            cam1_pose = rdp.transform.T_FLURDF
            cam2_pose = rdp.transform.T_FLURDF @ T_cam1_cam2
    
        # if T_ij in body frame
        else:
            cam1_pose = rdp.transform.T_FLURDF
            cam2_pose = T_cam1_cam2 @ rdp.transform.T_FLURDF

        cam1_trapezoid = self._get_camera_trapezoid_views(cam1_pose, cam1_params)
        cam2_trapezoid = self._get_camera_trapezoid_views(cam2_pose, cam2_params)
        return cam1_trapezoid.intersects(cam2_trapezoid)

    def _get_camera_trapezoid_views(self, cam_pose: np.ndarray, 
                                    cam_params: CameraParams) -> shapely.geometry.Polygon:
        trapezoid_corners_uv_depth = np.array([
            [0,                cam_params.height/2, self.params.cam_view_dist_bounds[0]],  
            [0,                cam_params.height/2, self.params.cam_view_dist_bounds[1]],
            [cam_params.width, cam_params.height/2, self.params.cam_view_dist_bounds[1]],
            [cam_params.width, cam_params.height/2, self.params.cam_view_dist_bounds[0]],
        ])
        trapezoid_cam = rdp.camera.pixel_depth_2_xyz(
            trapezoid_corners_uv_depth[:,0], 
            trapezoid_corners_uv_depth[:,1],
            trapezoid_corners_uv_depth[:,2],
            cam_params.K
        ).T

        trapezoid_world = rdp.transform.transform(cam_pose, trapezoid_cam)[:,:2]
        return shapely.geometry.Polygon(trapezoid_world)
    

def main():
    parser = argparse.ArgumentParser(description="Evaluate Submap Alignment Results")
    parser.add_argument("-s", "--eval-all-subdirs", nargs="+", type=str, default=[],
                        help="Evaluate all subdirectories in the given directories.")
    parser.add_argument("-i", "--input", type=str, nargs="+", action="append", default=[],
                        help="Input directory (should be root of the ROMAN demo output)." + 
                        "Optionally, include a name for the method." + 
                        "Example: -i /path/to/output1 -i /path/to/output2 cooloer method")
    parser.add_argument("-r", "--robots", nargs="+", required=True,  type=str,
                        help="Names of the robots involved in the alignment.")
    parser.add_argument("--rm-non-camera-overlap", action="store_true",
                        help="Remove alignments where there is no camera view overlap.")
    parser.add_argument("-d", "--eval-dist", type=float, default=10.0,
                        help="Distance threshold for evaluation (in meters).")
    args = parser.parse_args()

    eval_params = EvalParams(
        robot_names=args.robots,
        evaluation_distance_m=args.eval_dist,
        rm_non_camera_overlap=args.rm_non_camera_overlap,
    )

    for input_entry in args.input:
        assert len(input_entry) <= 2, \
            "Input entry must be a directory and optionally a name for the method."
    

    evaluator = SubmapAlignEvaluator(eval_params)

    eval_inputs = []
    for input_entry in args.input:
        dir_path = expandvars_recursive(input_entry[0])
        name = input_entry[1] if len(input_entry) == 2 else None
        eval_inputs.append(EvalInput(directory=dir_path, name=name))
    
    for base_dir in args.eval_all_subdirs:
        for subdir in os.listdir(base_dir):
            dir_path = expandvars_recursive(os.path.join(base_dir, subdir))
            if os.path.isdir(dir_path):
                eval_inputs.append(EvalInput(directory=dir_path))

    evaluator.load_results(eval_inputs)

    success_rates = {}
    for yaw_diff_label, (min_deg, max_deg) in STANDARD_YAW_DIFFS.items():
        success_rates[yaw_diff_label] = evaluator.evaluate_align_success_rate(
            yaw_diff_min_deg=min_deg,
            yaw_diff_max_deg=max_deg
        )

    for yaw_diff_label, rates in success_rates.items():
        print(f"\n=== Alignment Success Rates for Yaw Diff: {yaw_diff_label} ===")
        for name, rate in rates.items():
            print(f"{name}: {rate:.3f}")

if __name__ == "__main__":
    main()