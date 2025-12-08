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
from copy import deepcopy

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
    'timing_list'
]

STANDARD_YAW_DIFFS = {
    "all": (0.0, 180.0),
    "0 deg": (0.0, 60.0),
    "90 deg": (60.0, 120.0),
    "180 deg": (120.0, 180.0),
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
    precision_recall_sweep_use_num_assoc: bool = True
    precision_recall_sweep_held_submap_sim: float = 0.8
    precision_recall_sweep_held_num_assoc: int = 3
    precision_recall_sweep_submap_sim: Tuple[float, float] = (0.0, 1.0)
    precision_recall_sweep_num_assoc: Tuple[int, int] = (0, 20)
    place_rec_ref_is_single_robot: bool = False
    place_rec_require_pose_success: bool = False
    place_rec_rm_no_overlap_queries: bool = True
    place_rec_sweep_num_assoc: Tuple[int, int] = (0, 20)
    place_rec_overlap_dist: float = 30.0

    
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
    
@dataclass
class PR:
    precision: float
    recall: float

    @property
    def f1(self) -> float:
        if (
            self.precision + self.recall == 0 or
            np.isnan(self.precision) or np.isnan(self.recall)
        ):
            return float('nan')
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

@dataclass
class PRSweep:

    precisions: List[float]
    recalls: List[float]

    @classmethod
    def from_pr_list(cls, pr_list: List[PR]) -> 'PRSweep':
        precisions = [pr.precision for pr in pr_list]
        recalls = [pr.recall for pr in pr_list]
        return cls(precisions=precisions, recalls=recalls)
    
    def normalize(self):
        precision_list, recall_list = [], []
        for p, r in zip(self.precisions, self.recalls):
            if not np.isnan(p) and not np.isnan(r):
                precision_list.append(p)
                recall_list.append(r)

        if len(precision_list) == 0:
            self.precisions = []
            self.recalls = []
            return
        
        # assume that lists are sorted by an increasingly strict threshold
        # at the least strict, insert 0 precision, and the first recall
        precision_list.insert(0, 0.0)
        recall_list.insert(0, recall_list[0])
        # at the most strict, insert last precision, and 0 recall
        precision_list.append(precision_list[-1])
        recall_list.append(0.0)

        self.precisions = precision_list
        self.recalls = recall_list
        return

    def get_auc(self) -> float:
        self.normalize()
        if len(self.precisions) < 3:
            return float('nan')
        # reverse so that recalls is in ascending order for trapezoidal integration
        return np.trapz(self.precisions[::-1], self.recalls[::-1])

class SubmapAlignEvaluator():

    def __init__(self, params: EvalParams):
        assert params.place_rec_rm_no_overlap_queries, \
            "Only place_rec_rm_no_overlap_queries=True is supported."
        
        self.params = params
        self.results: Dict[str, SubmapAlignResults] = {}
        self.separate_pairs_results: Dict[str, Dict[Tuple[str, str], SubmapAlignResults]] = {}
        self.place_rec_sim_matrices: Dict[str, np.ndarray] = {}
        self.place_rec_submap_dist_matrices: Dict[str, np.ndarray] = {}
        self.robot_camera_params = {robot: None for robot in self.params.robot_names}

    def load_results(self, eval_inputs: Union[EvalInput, List[EvalInput]]):
        if isinstance(eval_inputs, EvalInput):
            eval_inputs = [eval_inputs]
        for eval_input in eval_inputs:
            result_paths = self._get_results_paths(eval_input)
            combined_results = None
            self.separate_pairs_results[eval_input.get_name()] = {}
            for path, robots in zip(result_paths, self.params.robot_pairs):
                new_result = SubmapAlignResults.load(path)
                self.separate_pairs_results[eval_input.get_name()][robots] = deepcopy(new_result)
                if self.params.rm_non_camera_overlap:
                    new_result = self._rm_non_camera_overlap(new_result, robots, eval_input)
 
                for matrix_attribute in ALIGN_RESULTS_MATRIX_ATTRIBUTES:
                    if hasattr(new_result, matrix_attribute):
                        matrix = getattr(new_result, matrix_attribute)
                        if matrix is None:
                            continue
                        setattr(new_result, matrix_attribute, np.array(matrix).reshape((-1)))

                if combined_results is None:
                    combined_results = new_result
                else:
                    for matrix_attribute in ALIGN_RESULTS_MATRIX_ATTRIBUTES:
                        if hasattr(new_result, matrix_attribute) and \
                                getattr(new_result, matrix_attribute) is not None:
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
    
    def evaluate_submap_precision_recall(
        self,
        submap_similarity_thresh: float = 0.5,
        num_associations_thresh: int = 3,
        use_submap_similarity: bool = True,
        use_num_associations: bool = True,
        names: List[str] = None,
    ) -> Dict[str, PR]:
        assert use_submap_similarity or use_num_associations, \
            "At least one of use_submap_similarity or use_num_associations must be True."
    
        precision_recall = {}

        all_results = self._get_named_results(names)
        
        for name, results in all_results.items():
            num_relevant = np.sum((results.robots_nearby_mat <= self.params.evaluation_distance_m))
            is_positive = np.ones_like(results.robots_nearby_mat, dtype=bool)
            if use_submap_similarity:
                if hasattr(results, 'similarity_mat'):
                    is_positive &= (results.similarity_mat >= submap_similarity_thresh)
                else:
                    print(f"Warning: Results for {name} do not have similarity_mat attribute.")
            if use_num_associations:
                is_positive &= (results.clipper_num_associations >= num_associations_thresh)
            
            correct_registration = (
                (results.clipper_angle_mat <= self.params.angular_err_thresh_deg) &
                (results.clipper_dist_mat <= self.params.distance_err_thresh_m)
            )
            true_positives = (
                (results.robots_nearby_mat <= self.params.evaluation_distance_m) &
                is_positive & correct_registration
            )
            not_same_place = (
                (results.robots_nearby_mat > self.params.evaluation_distance_m) |
                (np.isnan(results.robots_nearby_mat))
            )
            false_positives = (
                is_positive &
                (
                    # either the robots are farther away than thresh and the registration was wrong
                    (not_same_place & ~correct_registration) | 
                    # or they did overlap but the alignment was wrong
                    ~correct_registration
                )
            )

            num_false_positives = np.nansum(false_positives)
            num_true_positives = np.nansum(true_positives)
            precision = num_true_positives / (num_true_positives + num_false_positives) \
                if (num_true_positives + num_false_positives) > 0 else float('nan')
            recall = num_true_positives / num_relevant \
                if num_relevant > 0 else float('nan')
            precision_recall[name] = PR(precision=precision, recall=recall)
        return precision_recall
    
    def evaluate_precision_recall_sweep(self) -> Dict[str, PRSweep]:
        pr_sweep_results = {}
        if self.params.precision_recall_sweep_use_num_assoc:
            num_assoc_threshs = np.arange(
                self.params.precision_recall_sweep_num_assoc[0],
                self.params.precision_recall_sweep_num_assoc[1], + 1
            )
            submap_sim_threshs = np.array([
                self.params.precision_recall_sweep_held_submap_sim
                for _ in range(len(num_assoc_threshs))
            ])
        else:
            submap_sim_threshs = np.linspace(
                self.params.precision_recall_sweep_submap_sim[0],
                self.params.precision_recall_sweep_submap_sim[1],
                num=20
            )
            num_assoc_threshs = np.array([
                self.params.precision_recall_sweep_held_num_assoc
                for _ in range(len(submap_sim_threshs))
            ])

        for name, _ in self.results.items():

            prs = []
            for sim_thresh, assoc_thresh in zip(submap_sim_threshs, num_assoc_threshs):
                prs.append(self.evaluate_submap_precision_recall(
                    submap_similarity_thresh=sim_thresh,
                    num_associations_thresh=assoc_thresh,
                    names=[name]
                )[name])
            pr_sweep_results[name] = PRSweep.from_pr_list(prs)
        return pr_sweep_results
    
    def evaluate_place_recognition(
        self,
        num_associations_thresh: int = 3,
        names: List[str] = None,
    ) -> Dict[str, PR]:
        all_results = self._get_named_results(names)

        precision_recall = {}
        for name, results in all_results.items():
            sim_matrix, dist_matrix = self._get_sim_and_dist_matrix(name)
            
            eval_matrix = dist_matrix < self.params.evaluation_distance_m

            place_rec_succcess = None
            # TODO cache these matrices
            if self.params.place_rec_require_pose_success:
                dist_err = self._aggregate_multi_robot_matrix(
                    self._matrix_from_align_results(name, 'clipper_dist_mat'))
                ang_err = self._aggregate_multi_robot_matrix(
                    self._matrix_from_align_results(name, 'clipper_angle_mat'))
                place_rec_succcess = (
                    (ang_err <= self.params.angular_err_thresh_deg) &
                    (dist_err <= self.params.distance_err_thresh_m)
                )

            overlap_matrix = dist_matrix < self.params.place_rec_overlap_dist
            if self.params.place_rec_rm_no_overlap_queries:
                sim_matrix = sim_matrix[np.any(eval_matrix, axis=1),:][:,np.any(eval_matrix, axis=0)]
                if place_rec_succcess is not None:
                    place_rec_succcess = place_rec_succcess[
                        np.any(eval_matrix, axis=1),:][:,np.any(eval_matrix, axis=0)]
                overlap_matrix = overlap_matrix[np.any(eval_matrix, axis=1),:][:,np.any(eval_matrix, axis=0)]
                eval_matrix = eval_matrix[np.any(eval_matrix, axis=1),:][:,np.any(eval_matrix, axis=0)]
            assert overlap_matrix.shape[0] > 0, "No overlapping submaps"

            argmax_sim_i = np.nanargmax(sim_matrix, axis=1)
            max_sim_i = sim_matrix[np.arange(sim_matrix.shape[0]), argmax_sim_i]
            recall_at_0 = overlap_matrix[np.arange(overlap_matrix.shape[0]), argmax_sim_i].astype(np.bool_)
            if place_rec_succcess is not None:
                recall_at_0 = np.bitwise_and(recall_at_0, 
                    place_rec_succcess[np.arange(overlap_matrix.shape[0]), argmax_sim_i].astype(np.bool_))


            true_positives = np.sum(np.bitwise_and(max_sim_i >= num_associations_thresh, recall_at_0))
            false_positives = np.sum(np.bitwise_and(max_sim_i >= num_associations_thresh, np.bitwise_not(recall_at_0)))
            true_negatives = 0.0
            false_negatives = np.sum(np.bitwise_or(max_sim_i < num_associations_thresh, np.bitwise_not(recall_at_0)))

            precision = true_positives / (true_positives + false_positives) \
                if (true_positives + false_positives) > 0 else float('nan')
            recall = true_positives / (true_positives + false_negatives) \
                if (true_positives + false_negatives) > 0 else float('nan')
            precision_recall[name] = PR(precision=precision, recall=recall)

        return precision_recall
    
    def evaluate_place_recognition_sweep(self) -> Dict[str, PRSweep]:
        pr_sweep_results = {}
        num_assoc_threshs = np.arange(
            self.params.place_rec_sweep_num_assoc[0],
            self.params.place_rec_sweep_num_assoc[1], + 1
        )

        for name, _ in self.results.items():

            prs = []
            for assoc_thresh in num_assoc_threshs:
                prs.append(self.evaluate_place_recognition(
                    num_associations_thresh=assoc_thresh,
                    names=[name]
                )[name])
            pr_sweep_results[name] = PRSweep.from_pr_list(prs)
        return pr_sweep_results
    
    def evaluate_timing(self) -> Dict[str, float]:
        timing_results = {}
        for name, results in self.results.items():
            if results.timing_list is None:
                timing_results[name] = float('nan')
                continue
            mean_time = np.nanmean(results.timing_list)
            timing_results[name] = mean_time
        return timing_results
               
    def plot_precision_recall_sweep(
        self,
        pr_sweeps: Dict[str, PRSweep],
        fig: plt.Figure = None,
        ax: plt.Axes = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        if fig is None:
            fig = plt.gcf()
        if ax is None:
            ax = plt.gca()

        for name, pr_sweep in pr_sweeps.items():
            ax.plot(pr_sweep.recalls, pr_sweep.precisions, label=name)
            
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")

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

    def _get_named_results(self, names: List[str] = None) -> Dict[str, SubmapAlignResults]:
        if names is None:
            return self.results
        else:
            return {name: self.results[name] for name in names if name in self.results}

    def _get_sim_and_dist_matrix(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct a single similarity matrix and submap distance matrix. The rows
            of the matrices correspond to submaps from all robots concatenated together,
            and the columns correspond to submaps from all robots concatenated together or 
            a single robot at a time (depending on self.params.place_rec_ref_is_single_robot).

        Args:
            name (str): Result name for which the matrices should be constructed.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Similarity matrix (in terms of number 
                of associations) and submap distance matrix.
        """
        if name in self.place_rec_sim_matrices and \
              name in self.place_rec_submap_dist_matrices:
            return (self.place_rec_sim_matrices[name],
                    self.place_rec_submap_dist_matrices[name])

        sim_matrix = self._matrix_from_align_results(eval_name=name, 
                                                     matrix_name='clipper_num_associations')
        submap_dist_matrix = self._matrix_from_align_results(eval_name=name,
                                                            matrix_name='robots_nearby_mat')
        
        for i in range(len(self.params.robot_names)):
            if sim_matrix[i][i] is None:
                size_i = sim_matrix[i][i-1].shape[0]
                sim_matrix[i][i] = np.zeros((size_i, size_i)) * np.nan
                submap_dist_matrix[i][i] = np.zeros((size_i, size_i)) * np.nan

        combined_sim_matrix = self._aggregate_multi_robot_matrix(sim_matrix)
        combined_submap_dist_matrix = self._aggregate_multi_robot_matrix(submap_dist_matrix)
        self.place_rec_sim_matrices[name] = combined_sim_matrix
        self.place_rec_submap_dist_matrices[name] = combined_submap_dist_matrix
        return combined_sim_matrix, combined_submap_dist_matrix
    
    def _matrix_from_align_results(self, eval_name: str, matrix_name: str):
        matrix = [[None for _ in range(len(self.params.robot_names))] for _ in range(len(self.params.robot_names))]
        for i, robot_i in enumerate(self.params.robot_names):
            for j, robot_j in enumerate(self.params.robot_names):
                if i == j:
                    try:
                        matrix[i][j] = np.nan * \
                            self.separate_pairs_results[eval_name][(robot_i, robot_j)].__getattribute__(matrix_name)
                    except KeyError as e:
                        matrix[i][j] = None
                elif j < i:
                    matrix[i][j] = matrix[j][i].T
                else:
                    matrix[i][j] = \
                        self.separate_pairs_results[eval_name][(robot_i, robot_j)].__getattribute__(matrix_name)
        for i in range(len(self.params.robot_names)):
            if matrix[i][i] is None:
                size_i = matrix[i][i-1].shape[0]
                matrix[i][i] = np.zeros((size_i, size_i)) * np.nan
        return matrix
    
    def _aggregate_multi_robot_matrix(self, matrix: List[List[np.array]]):
        if not self.params.place_rec_ref_is_single_robot:
            matrix_combined = np.concatenate([
                np.concatenate(matrix[i], axis=1) for i in range(len(matrix))\
            ], axis=0)
            return matrix_combined
        else:
            num_robots = len(matrix)
            flattened_sim_matrix = [x for xs in matrix for x in xs]
            max_num_submaps = np.max([x.shape[0] for x in flattened_sim_matrix])
            matrix_combined = np.zeros((max_num_submaps*num_robots**2, max_num_submaps))*np.nan

            for i in range(num_robots):
                for j in range(num_robots):
                    idx_i = i*max_num_submaps*num_robots + j*max_num_submaps
                    matrix_combined[idx_i:idx_i + matrix[i][j].shape[0],
                                    :matrix[i][j].shape[1]] = matrix[i][j]
            return matrix_combined
        
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
    parser.add_argument("--plot-precision-recall", type=str, default=None,
                        help="If set, saves precision-recall curves to the given image file.")
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
    precision_recall_sweeps = evaluator.evaluate_precision_recall_sweep()
    place_rec_pr_sweeps = evaluator.evaluate_place_recognition_sweep()
    timing_results = evaluator.evaluate_timing()

    for yaw_diff_label, rates in success_rates.items():
        print(f"\n=== Alignment Success Rates for Yaw Diff: {yaw_diff_label} ===")
        for name, rate in rates.items():
            print(f"{name}: {rate:.3f}")

    print(f"\n=== Precision-Recall AUC Results ===")
    for name, pr_sweep in precision_recall_sweeps.items():
        print(f"{name}: {pr_sweep.get_auc():.3f}")

    print(f"\n=== Place Recognition Precision-Recall AUC Results ===")
    for name, pr_sweep in place_rec_pr_sweeps.items():
        print(f"{name}: {pr_sweep.get_auc():.3f}")

    print(f"\n=== Timing Results (ms) ===")
    for name, time in timing_results.items():
        print(f"{name}: {time*1e3:.1f}")

    if args.plot_precision_recall is not None:
        output_file = expandvars_recursive(args.plot_precision_recall)
        fig, ax = plt.subplots()
        evaluator.plot_precision_recall_sweep(precision_recall_sweeps, fig, ax)
        plt.legend()
        plt.savefig(output_file)

if __name__ == "__main__":
    main()