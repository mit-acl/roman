import numpy as np
from numpy.linalg import inv
from typing import List, Dict
from dataclasses import dataclass
import gtsam
from tqdm import tqdm

from robotdatapy.data.pose_data import PoseData
from robotdatapy.exceptions import NoDataNearTimeException
from robotdatapy import transform

from g2o_and_time_to_pose_data import g2o_and_time_to_pose_data, time_vertex_mapping


# class LoopClosure:
@dataclass
class LoopClosure:
    vertex0: int
    vertex1: int
    vertex0_time: float
    vertex1_time: float
    xyz_quat: np.ndarray
    information: np.ndarray

    def vertex(self, robot_num: int) -> int:
        assert robot_num == 0 or robot_num == 1, "robot_num must be 0 or 1"
        return self.vertex0 if robot_num == 0 else self.vertex1
    
    def vertex_time(self, robot_num: int) -> float:
        assert robot_num == 0 or robot_num == 1, "robot_num must be 0 or 1"
        return self.vertex0_time if robot_num == 0 else self.vertex1_time

    def robot_id(self, robot_num: int) -> str:
        return chr(gtsam.Symbol(self.vertex(robot_num)).chr())
    
    def transform(self) -> np.ndarray:
        return transform.xyz_quat_to_transform(self.xyz_quat[:3], self.xyz_quat[3:])
    
    def __str__(self):
        return f"EDGE_SE3:QUAT {self.vertex0} {self.vertex1} " + \
               " ".join([str(x) for x in self.xyz_quat]) + " " + \
               " ".join([str(x) for x in self.information])

def extract_additional_lc(
    loop_closures: List[LoopClosure], 
    pd_ref: Dict[str, PoseData], 
    pd_elc: Dict[str, PoseData],
    tv_ref: Dict[str, Dict[float, int]]
) -> List[LoopClosure]:
    """
    Transforms loop closures from one set of timestamps to another set of timestamps.
    Includes handling of additional transformation that must happen to express loop closures
    in the frame of the reference pose graph.

    Args:
        loop_closures (List[LoopClosure]): list of additional loop closures
        pd_ref (Dict[str, PoseData]): pose data for the reference pose graph
        pd_elc (Dict[str, PoseData]): pose data for the pose graph with extra loop closures
        tv_ref (Dict[str, Dict[float, int]]): mapping from timestamp to vertex index for the 
            reference pose graph for each robot

    Returns:
        List[LoopClosure]: list of extra loop closures to add to the reference pose graph
    """
    extra_lc = []
    for pd in list(pd_ref.values()) + list(pd_elc.values()):
        pd.interp = True
        pd.time_tol = 700.0
    for pd in pd_ref.values():
        pd.times = np.array(pd.times[1:])
        pd.positions = pd.positions[1:]
        pd.orientations = pd.orientations[1:]

    for lc in tqdm(loop_closures):
        # step 3a: for each loop closure in the second g2o file, find the nearest timestamped 
        # vertex from the first g2o file. 
        vxs_ref = []
        times_ref = []
        T_t0_tnear = [] # transformation from reference time to extra_lc time

        for i in range(2):
            robot = lc.robot_id(i)
            t0 = lc.vertex_time(i)
            # try:
            t_near = pd_ref[robot].nearest_time(t0) # nearest time in reference
            times_ref.append(t_near)
            vxs_ref.append(tv_ref[robot][t_near])

            # 3b: Find the transform between the timestamp from the second g2o file and the nearest
            # vertex from the first g2o file (could average from the two PoseData objects)

            try:
                T_odom_t0_e = pd_elc[robot].pose(t0)
            except ValueError as e:
                # problem with vertex 0 and 1 having same timestamp (data problem, handle here)
                if set(pd_elc[robot].idx(t0)) == set([0, 1]):
                    position = pd_elc[robot].positions[0]
                    orientation = pd_elc[robot].orientations[0]
                    T_odom_t0_e = transform.xyz_quat_to_transform(position, orientation)
                else:
                    raise e

            # use only extra loop closure data to get the transform since this is likely
            # a finer pose representation (in terms of time)
            T_odom_tnear_e = pd_elc[robot].pose(t_near)
            # T_odom_t0_r = pd_ref[robot].pose(t0)
            # T_odom_tnear_r = pd_ref[robot].pose(t_near)

            T_t0_tnear_elc = inv(T_odom_t0_e) @ T_odom_tnear_e
            # T_t0_tnear_ref = inv(T_odom_t0_r) @ T_odom_tnear_r

            # T_t0_tnear.append(transform.mean([
            #     T_t0_tnear_elc, T_t0_tnear_ref
            # ]))

            T_t0_tnear.append(T_t0_tnear_elc)

        # 3c: Transform the loop closure into the frame of the first g2o file
        T_p0e_p1e = lc.transform() # pose of vertex 1 in the frame of vertex 0
        T_p0e_p0r = T_t0_tnear[0] # transformation from vertex 0 in the reference 
                                  # time to vertex 0 in the extra_lc time
        T_p1e_p1r = T_t0_tnear[1] # transformation from vertex 1 in the reference
                                  # time to vertex 1 in the extra_lc time
        T_p0r_p1r = inv(T_p0e_p0r) @ T_p0e_p1e @ T_p1e_p1r

        # 3d: Add new loop closure object
        extra_lc.append(LoopClosure(
            vertex0=vxs_ref[0],
            vertex1=vxs_ref[1],
            vertex0_time=times_ref[0],
            vertex1_time=times_ref[1],
            xyz_quat=transform.transform_to_xyz_quat(T_p0r_p1r),
            information=lc.information
        ))

    return extra_lc

def combine_visual_object_lc(
    g2o_reference: str, 
    g2o_extra_lc: str, 
    vertex_times_reference: str, 
    vertex_times_extra_lc: str
) -> List[str]:
    """
    Combine two g2o files with timestamps into one g2o file with additional loop closures.

    Args:
        g2o_reference (str): Path to main g2o file to which additional loop closures will be added.
        g2o_extra_lc (str): Path to g2o file with additional loop closures.
        vertex_times_reference (str): Path to the file containing timestamps for the vertices in the main g2o file.
        vertex_times_extra_lc (str): Path to the file containing timestamps for the vertices in the g2o file with additional loop closures.

    Returns:
        List[str]: List of lines in the new g2o file.
    """

    # step 0: open files
    with open(g2o_reference, 'r') as f:
        g2o_lines_ref = f.readlines()
        g2o_lines_split_ref = [line.strip().split() for line in g2o_lines_ref]
    with open(g2o_extra_lc, 'r') as f:
        g2o_lines_elc = f.readlines()
        g2o_lines_split_elc = [line.strip().split() for line in g2o_lines_elc]
    # with open(vertex_times_reference, 'r') as f:
    #     times_lines_ref = f.readlines()
    # with open(vertex_times_extra_lc, 'r') as f:
    #     times_lines_elc = f.readlines()
    
    # step 1: get a list of robots
    robot_symbols = set([chr(gtsam.Symbol(int(line[1])).chr()) 
                         for line in g2o_lines_split_ref if line[0] == 'VERTEX_SE3:QUAT'])

    # step 2: Extract data
    # 2a: Create PoseData for each g2o file and for each robot
    pd_ref = dict()
    pd_elc = dict()
    for robot_id in robot_symbols:
        pd_ref[robot_id] = g2o_and_time_to_pose_data(g2o_reference, vertex_times_reference, ord(robot_id) - ord('a'))
        pd_elc[robot_id] = g2o_and_time_to_pose_data(g2o_extra_lc, vertex_times_extra_lc, ord(robot_id) - ord('a'))

    # 2b: Get a mapping from vertex index to timestamp for each robot
    vt_ref = time_vertex_mapping(vertex_times_reference, use_gtsam_idx=True)
    vt_elc = time_vertex_mapping(vertex_times_extra_lc, use_gtsam_idx=True)
    tv_ref = {r: dict() for r in robot_symbols}
    for v, t in vt_ref.items():
        robot = chr(gtsam.Symbol(v).chr())
        tv_ref[robot][t] = v

    # 2c: Extract loop closures from the second g2o file
    loop_closures = []
    for line in g2o_lines_split_elc:
        if line[0] == 'EDGE_SE3:QUAT':
            vertex0 = int(line[1])
            vertex1 = int(line[2])
            if np.abs(vertex0 - vertex1) == 1:
                continue
            vertex0_time = vt_elc[vertex0]
            vertex1_time = vt_elc[vertex1]
            xyz_quat = np.array([float(x) for x in line[3:10]])
            information = np.array([float(x) for x in line[10:]])
            loop_closures.append(LoopClosure(
                vertex0, vertex1, vertex0_time, vertex1_time, xyz_quat, information))
        
    # step 3: attach each loop closure from the "extra_lc" g2o file to two
    # vertices in the "reference" g2o file
    extra_lc = extract_additional_lc(loop_closures, pd_ref, pd_elc, tv_ref)

    # step 4: return the new g2o file lines
    return g2o_lines_ref + ["# NEW LOOP CLOSURES"] + [str(lc) for lc in extra_lc]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('ref_g2o', type=str, help='Path to reference g2o file.')
    parser.add_argument('extra_lc_g2o', type=str, help='Path to g2o file with extra loop closures.')
    parser.add_argument('ref_time', type=str, help='Path to file with vertex times for reference g2o file.')
    parser.add_argument('extra_lc_time', type=str, help='Path to file with vertex times for g2o file with extra loop closures.')
    parser.add_argument('-o', '--output', type=str, default=None, help='Path to output g2o file.')
    args = parser.parse_args()

    new_g2o_lines = combine_visual_object_lc(
        args.ref_g2o, args.extra_lc_g2o, args.ref_time, args.extra_lc_time
    )

    if args.output is not None:
        with open(args.output, 'w') as f:
            for line in new_g2o_lines:
                f.write(line.strip() + '\n')
            f.close()