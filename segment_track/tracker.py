import numpy as np
from typing import List
from dataclasses import dataclass

from robot_utils.robot_data.img_data import CameraParams

from segment_track.segment import Segment
from segment_track.observation import Observation
from segment_track.global_nearest_neighbor import global_nearest_neighbor

@dataclass
class Cylinder():
    """
    A cylinder object
    """
    position: np.array
    radius: float
    height: float
    

class Tracker():

    def __init__(
            self,
            camera_params: CameraParams,
            pixel_std_dev: float,
            min_iou: float,
            min_sightings: int,
            max_t_no_sightings: int,
            merge_objects: bool = False,
            merge_objects_iou: float = 0.25
    ):
        self.camera_params = camera_params
        self.pixel_std_dev = pixel_std_dev
        self.min_iou = min_iou
        self.min_sightings = min_sightings
        self.max_t_no_sightings = max_t_no_sightings
        self.merge_objects = merge_objects
        self.merge_objects_iou = merge_objects_iou

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
    
    def filter_segment_graveyard(self, rm_fun: callable):
        """
        Filter the segment graveyard when rm_fun returns True
        """
        self.segment_graveyard = [seg for seg in self.segment_graveyard if not rm_fun(seg)]
        return
    
    def merge(self):
        """
        Merge segments with high overlap
        """
        max_iter = 100
        n = 0
        edited = True

        to_delete = []
        for seg in self.segment_graveyard:
            try:
                seg.reconstruction3D(width_height=True)
            except:
                to_delete.append(seg)
        for seg in to_delete:
            self.segment_graveyard.remove(seg)

        # repeatedly try to merge until no further merges are possible
        while n < max_iter and edited:
            edited = False
            n += 1

            for i, seg1 in enumerate(self.segment_graveyard):
                for j, seg2 in enumerate(self.segment_graveyard):
                    if i >= j:
                        continue
                    reconstruction = seg1.reconstruction3D(width_height=True)
                    c1 = Cylinder(reconstruction[:3], 0.5*reconstruction[3], reconstruction[4])
                    reconstruction = seg2.reconstruction3D(width_height=True)
                    c2 = Cylinder(reconstruction[:3], 0.5*reconstruction[3], reconstruction[4])
                    intersection = self.cylinder_intersection(c1, c2)
                    combined_vol = c1.height * np.pi * c1.radius**2 + \
                        c2.height * np.pi * c2.radius**2
                    iou = intersection / (combined_vol - intersection)
                    if iou > self.merge_objects_iou:
                        for obs in seg2.observations:
                            seg1.update(obs)
                        self.segment_graveyard.pop(j)
                        edited = True
                        break
                if edited:
                    break
        return

    def cylinder_intersection(self, c1: Cylinder, c2: Cylinder):
        """
        Compute the intersection volume between two cylinders
        """
        d = np.linalg.norm(c1.position[:2] - c2.position[:2])
        if np.abs(c1.position[2] - c2.position[2]) > 0.5 * (c1.height + c2.height) \
            or d > c1.radius + c2.radius: # no intersection
            return 0.0
        intersecting_height = \
            (np.min([c1.position[2] + 0.5*c1.height, c2.position[2] + 0.5*c2.height]) # top
            - np.max([c1.position[2] - 0.5*c1.height, c2.position[2] - 0.5*c2.height])) # bottom
        if d <= np.abs(c1.radius - c2.radius): # one cylinder is inside the other
            return np.pi * np.min([c1.radius, c2.radius])**2 * intersecting_height
        r1 = c1.radius
        r2 = c2.radius
        return (r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1)) \
            + r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2)) \
            - 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))) \
            * intersecting_height
            