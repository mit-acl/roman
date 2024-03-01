import numpy as np
from typing import List

from robot_utils.robot_data.img_data import CameraParams

from segment_track.segment import Segment
from segment_track.observation import Observation
from segment_track.global_nearest_neighbor import global_nearest_neighbor

class Tracker():

    def __init__(
            self,
            camera_params: CameraParams,
            pixel_std_dev: float,
            min_iou: float,
            min_sightings: int,
            max_t_no_sightings: int
    ):
        self.camera_params = camera_params
        self.pixel_std_dev = pixel_std_dev
        self.min_iou = min_iou
        self.min_sightings = min_sightings
        self.max_t_no_sightings = max_t_no_sightings

        self.segment_nursery = []
        self.segments = []
        self.segment_graveyard = []
        self.id_counter = 0

    def update(self, t: float, observations: List[Observation]):
        
        # associate observations with segments
        associated_pairs = global_nearest_neighbor(
            self.segments + self.segment_nursery, observations, self.mask_similarity, self.min_iou
        )

        # separate segments associated with nursery and normal segments
        pairs_existing = [[seg_idx, obs_idx] for seg_idx, obs_idx \
                                in associated_pairs if seg_idx < len(self.segments)]
        pairs_nursery = [[seg_idx - len(self.segments), obs_idx] for seg_idx, obs_idx \
                                in associated_pairs if seg_idx >= len(self.segments)]

        # update segments with associated observations
        for seg_idx, obs_idx in pairs_existing:
            self.segments[seg_idx].update(observations[obs_idx])
        for seg_idx, obs_idx in pairs_nursery:
            self.segment_nursery[seg_idx].update(observations[obs_idx])

        # handle moving existing segments to graveyard
        to_rm = [seg for seg in self.segments \
                    if t - seg.last_seen > self.max_t_no_sightings]
        for seg in to_rm:
            self.segment_graveyard.append(seg)
            self.segments.remove(seg)

        to_rm = [seg for seg in self.segment_nursery \
                    if t - seg.last_seen > self.max_t_no_sightings]
        for seg in to_rm:
            self.segment_nursery.remove(seg)

        # handle moving segments from nursery to normal segments
        to_upgrade = [seg for seg in self.segment_nursery \
                        if seg.num_sightings >= self.min_sightings]
        for seg in to_upgrade:
            self.segment_nursery.remove(seg)
            self.segments.append(seg)

        # add new segments
        associated_obs = [obs_idx for _, obs_idx in associated_pairs]
        new_observations = [obs for idx, obs in enumerate(observations) \
                            if idx not in associated_obs]
        for obs in new_observations:
            self.segment_nursery.append(
                Segment(obs, self.camera_params, self.pixel_std_dev, self.id_counter))
            self.id_counter += 1
            
        return

    def mask_similarity(self, segment: Segment, observation: Observation):
        """
        Compute the similarity between the mask of a segment and an observation
        """
        if segment.last_mask is None:
            return 0
        intersection = np.logical_and(segment.last_mask, observation.mask).sum()
        union = np.logical_or(segment.last_mask, observation.mask).sum()
        return intersection / union
