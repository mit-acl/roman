###########################################################
#
# mapper.py
#
# ROMAN open-set segment mapper class
#
# Authors: Mason Peterson, Yulun Tian, Lucas Jia, Qingyuan Li
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import cv2 as cv
import open3d as o3d

from robotdatapy.data.img_data import CameraParams

from roman.object.similiarity_metrics import ChamferDistance
from roman.object.segment import Segment
from roman.map.observation import Observation
from roman.map.global_nearest_neighbor import global_nearest_neighbor
from roman.map.map import ROMANMap
from roman.params.mapper_params import MapperParams

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Mapper():

    def __init__(self, params: MapperParams, camera_params: CameraParams):
        self.params = params
        self.camera_params = camera_params

        self.segment_nursery = []
        self.segments = []
        self.inactive_segments = []
        self.segment_graveyard = []
        self.id_counter = 0
        self.last_pose = None
        self.poses_flu_history = []
        self.times_history = []
        self._T_camera_flu = np.eye(4)

    def update(self, t: float, pose: np.array, observations: List[Observation]):

        # have T_WC, want T_WB
        # T_WB = T_WC @ T_CB
        self.poses_flu_history.append(pose @ self._T_camera_flu)
        self.times_history.append(t)
        
        # store last pose
        self.last_pose = pose.copy()
        
        # associate observations with segments
        # mask_similarity = lambda seg, obs: max(self.mask_similarity(seg, obs, projected=False), 
        #                                        self.mask_similarity(seg, obs, projected=True))
        associated_pairs = global_nearest_neighbor(
            self.segments + self.segment_nursery, observations, 
            self.chamfer_distance_similarity if self.params.association_method == 'chamfer' else self.voxel_grid_similarity, 
            # negate: see chamfer_distance_similarity
            -self.params.max_chamfer_dist if self.params.association_method == 'chamfer' else self.params.min_iou, 
        )

        # separate segments associated with nursery and normal segments
        pairs_existing = [[seg_idx, obs_idx] for seg_idx, obs_idx \
                                in associated_pairs if seg_idx < len(self.segments)]
        pairs_nursery = [[seg_idx - len(self.segments), obs_idx] for seg_idx, obs_idx \
                                in associated_pairs if seg_idx >= len(self.segments)]

        # update segments with associated observations
        for seg_idx, obs_idx in pairs_existing:
            self.segments[seg_idx].update(observations[obs_idx], integrate_points=True)
            # if self.segments[seg_idx].num_points == 0:
            #     self.segments.pop(seg_idx)
        for seg_idx, obs_idx in pairs_nursery:
            # forcing add does not try to reconstruct the segment
            self.segment_nursery[seg_idx].update(observations[obs_idx], integrate_points=True)
            # if self.segment_nursery[seg_idx].num_points == 0:
            #     self.segment_nursery.pop(seg_idx)

        # delete masks for segments that were not seen in this frame
        for seg in self.segments:
            if not np.allclose(t, seg.last_seen, rtol=0.0):
                seg.last_observation.mask = None

        # handle moving existing segments to inactive
        to_rm = [seg for seg in self.segments \
                    if t - seg.last_seen > self.params.max_t_no_sightings \
                        or seg.num_points == 0]
        for seg in to_rm:
            if seg.num_points == 0:
                self.segments.remove(seg)
                continue
            try:
                seg.final_cleanup(epsilon=self.params.segment_voxel_size*5.0)
                self.inactive_segments.append(seg)
                self.segments.remove(seg)
            except: # too few points to form clusters
                self.segments.remove(seg)
            
        # handle moving inactive segments to graveyard
        to_rm = [seg for seg in self.inactive_segments \
                    if t - seg.last_seen > self.params.segment_graveyard_time \
                        or np.linalg.norm(seg.last_observation.pose[:3,3] - pose[:3,3]) \
                            > self.params.segment_graveyard_dist]
        for seg in to_rm:
            self.segment_graveyard.append(seg)
            self.inactive_segments.remove(seg)

        to_rm = [seg for seg in self.segment_nursery \
                    if t - seg.last_seen > self.params.max_t_no_sightings \
                        or seg.num_points == 0]
        for seg in to_rm:
            self.segment_nursery.remove(seg)

        # handle moving segments from nursery to normal segments
        to_upgrade = [seg for seg in self.segment_nursery \
                        if seg.num_sightings >= self.params.min_sightings]
        for seg in to_upgrade:
            self.segment_nursery.remove(seg)
            self.segments.append(seg)

        # add new segments
        associated_obs = [obs_idx for _, obs_idx in associated_pairs]
        new_observations = [obs for idx, obs in enumerate(observations) \
                            if idx not in associated_obs]
        for obs in new_observations:
            new_seg = Segment(obs, self.camera_params, self.id_counter, self.params.segment_voxel_size)
            if new_seg.num_points == 0: # guard from observations coming in with no points
                continue
            self.segment_nursery.append(new_seg)
            self.id_counter += 1

        self.merge()
            
        return
    
    def voxel_grid_similarity(self, segment: Segment, observation: Observation):
        """
        Compute the similarity between the voxel grids of a segment and an observation
        """
        voxel_size = self.params.iou_voxel_size
        segment_voxel_grid = segment.get_voxel_grid(voxel_size)
        observation_voxel_grid = observation.get_voxel_grid(voxel_size)
        return segment_voxel_grid.iou(observation_voxel_grid)
    
    def chamfer_distance_similarity(self, segment: Segment, observation: Observation):
        """
        Compute the similarity between a segment and observation using their chamfer distance
        """
        # Negate because smaller distance is larger similarity
        return -ChamferDistance.chamfer_distance(segment.pcd, observation.pcd)

    def mask_similarity(self, segment: Segment, observation: Observation, projected: bool = False):
        """
        Compute the similarity between the mask of a segment and an observation
        """
        if not projected or segment in self.segment_nursery:
            segment_propagated_mask = segment.last_observation.mask_downsampled
            # segment_propagated_mask = segment.propagated_last_mask(observation.time, observation.pose, downsample_factor=self.mask_downsample_factor)
            if segment_propagated_mask is None:
                iou = 0.0
            else:
                iou = Mapper.compute_iou(segment_propagated_mask, observation.mask_downsampled)

        # compute the similarity using the projected mask rather than last mask
        else:
            segment_mask = segment.reconstruct_mask(observation.pose, 
                            downsample_factor=self.params.mask_downsample_factor)
            iou = Mapper.compute_iou(segment_mask, observation.mask_downsampled)
        return iou
    
    @staticmethod
    def compute_iou(mask1, mask2):
        """Compute the intersection over union (IoU) of two masks.

        Args:
            mask1 (_type_): _description_
            mask2 (_type_): _description_
        """

        assert mask1.shape == mask2.shape
        logger.debug(f"Compute IoU for shape {mask1.shape}")
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if np.isclose(union, 0):
            return 0.0
        return float(intersection) / float(union)

    def remove_bad_segments(self, segments: List[Segment], min_volume: float=0.0, min_max_extent: float=0.0, plane_prune_params: List[float]=[np.inf, np.inf, 0.0]):
        """
        Remove segments that have small volumes or have no points

        Args:
            segments (List[Segment]): List of segments
            min_volume (float, optional): Minimum allowable segment volume. Defaults to 0.0.

        Returns:
            segments (List[Segment]): Filtered list of segments
        """
        to_delete = []
        # reason = []
        for seg in segments:
            try:
                extent = np.sort(seg.extent) # in ascending order
                if seg.num_points == 0:
                    to_delete.append(seg)
                    # reason.append(f"Segment {seg.id} has no points")
                elif seg.volume < min_volume:
                    to_delete.append(seg)
                    # reason.append(f"Segment {seg.id} has volume {seg.volume} < {min_volume}")
                elif extent[-1] < min_max_extent:
                    to_delete.append(seg)
                    # reason.append(f"Segment {seg.id} has max extent {np.max(seg.extent)} < {min_max_extent}"
                elif extent[2] > plane_prune_params[0] and extent[1] > plane_prune_params[1] and extent[0] < plane_prune_params[2]:
                    to_delete.append(seg)
                    # reason.append(f"Segment {seg.id} has extent {seg.extent} which is likely a plane")
            except: 
                to_delete.append(seg)
                # reason.append(f"Segment {seg.id} has an error in extent/volume computation")
        for seg in to_delete:
            segments.remove(seg)
            # for r in reason:
            #     print(r)
        return segments
    
    def merge(self):
        """
        Merge segments with high overlap
        """

        # Right now existing segments are merged with other existing segments or 
        # segments inthe graveyard. Heuristic for merging involves either projected IOU or 
        # 3D IOU. Should look into more.

        max_iter = 100
        n = 0
        edited = True

        self.inactive_segments = self.remove_bad_segments(
            self.inactive_segments, 
            min_max_extent=self.params.min_max_extent, 
            plane_prune_params=self.params.plane_prune_params
        )
        self.segments = self.remove_bad_segments(self.segments)

        # repeatedly try to merge until no further merges are possible
        while n < max_iter and edited:
            edited = False
            n += 1

            for i, seg1 in enumerate(self.segments):
                for j, seg2 in enumerate(self.segments + self.inactive_segments):
                    if i >= j:
                        continue

                    # if segments are very far away, don't worry about doing extra checking
                    if np.mean(seg1.points) - np.mean(seg2.points) > \
                        .5 * (np.max(seg1.extent) + np.max(seg2.extent)):
                        continue 

                    if self.params.merge_method == 'chamfer': # Chamfer distance-based merging
                        chamfer_dist = ChamferDistance.chamfer_distance(seg1.pcd, seg2.pcd)

                        merge_flag = chamfer_dist < self.params.merge_objects_max_chamfer_dist
                    else: # IOU-based merging
                        maks1 = seg1.reconstruct_mask(self.last_pose)
                        maks2 = seg2.reconstruct_mask(self.last_pose)
                        intersection2d = np.logical_and(maks1, maks2).sum()
                        union2d = np.logical_or(maks1, maks2).sum()
                        iou2d = intersection2d / union2d

                        iou3d = seg1.get_voxel_grid(self.params.iou_voxel_size).iou(
                            seg2.get_voxel_grid(self.params.iou_voxel_size))
                        
                        merge_flag = iou3d > self.params.merge_objects_iou_3d or iou2d > self.params.merge_objects_iou_2d

                    # print(f"chamfer: {ChamferDistance.chamfer_distance(seg1.pcd, seg2.pcd)}. iou3d: {iou3d}. iou2d: {iou2d}")

                    if merge_flag:
                        seg1.update_from_segment(seg2)
                        seg1.id = min(seg1.id, seg2.id)
                        if seg1.num_points == 0:
                            self.segments.pop(i)
                        elif j < len(self.segments):
                            self.segments.pop(j)
                        else:
                            self.inactive_segments.pop(j - len(self.segments))
                        edited = True
                        break
                if edited:
                    break
        return
            
    def make_pickle_compatible(self):
        """
        Make the Mapper object pickle compatible
        """
        for seg in self.segments + self.segment_nursery + self.inactive_segments + self.segment_graveyard:
            seg.reset_memoized()
        return
    
    def get_segment_map(self) -> List[Segment]:
        """
        Get the segment map
        """
        segment_map = self.remove_bad_segments(
            self.segment_graveyard + self.inactive_segments + 
            self.segments)
        for seg in segment_map:
            seg.reset_memoized()
        return segment_map
    
    def get_roman_map(self) -> ROMANMap:
        """
        Return the full ROMAN map.

        Returns:
            ROMANMap: Map of objects
        """
        segment_map = self.get_segment_map()
        return ROMANMap(
            segments=segment_map,
            trajectory=self.poses_flu_history,
            times=self.times_history,
            poses_are_flu=True
        )
    
    def set_T_camera_flu(self, T_camera_flu: np.array):
        """
        Set the transformation matrix from camera frame to forward-left-up frame
        """
        self._T_camera_flu = T_camera_flu
        return
    
    @property
    def T_camera_flu(self):
        return self._T_camera_flu
            