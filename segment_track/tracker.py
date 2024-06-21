import numpy as np
from typing import List
from dataclasses import dataclass
import cv2 as cv
import open3d as o3d

from robot_utils.robot_data.img_data import CameraParams

from segment_track.segment import Segment
from segment_track.observation import Observation
from segment_track.global_nearest_neighbor import global_nearest_neighbor

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
            merge_objects_iou: float = 0.25,
            max_cov_axis: float = 2.0,
            mask_downsample_factor: int = 1,
            min_max_extent: float = 0.5, # to get rid of very small objects
            plane_prune_params: List[float] = [3.0, 3.0, 0.5] # to get rid of planar objects (likely background)
    ):
        self.camera_params = camera_params
        self.pixel_std_dev = pixel_std_dev
        self.min_iou = min_iou
        self.min_sightings = min_sightings
        self.max_t_no_sightings = max_t_no_sightings
        self.min_orb_matches = 5
        self.num_orb_features = 10
        self.merge_objects = merge_objects
        self.merge_objects_iou_3d = merge_objects_iou
        self.merge_objects_iou_2d = 0.8
        self.max_cov_axis = max_cov_axis
        self.mask_downsample_factor = mask_downsample_factor
        # self.min_volume = .25**3 # 25x25x25 cm^3
        self.min_max_extent = min_max_extent
        self.plane_prune_params = plane_prune_params

        self.segment_nursery = []
        self.segments = []
        self.segment_graveyard = []
        self.id_counter = 0
        self.last_pose = None

    def update(self, t: float, pose: np.array, observations: List[Observation]):
        
        # store last pose
        self.last_pose = pose.copy()
        
        # associate observations with segments
        mask_similarity = lambda seg, obs: max(self.mask_similarity(seg, obs, projected=False), 
                                               self.mask_similarity(seg, obs, projected=True))
        associated_pairs = global_nearest_neighbor(
            self.segments + self.segment_nursery, observations, mask_similarity, self.min_iou
        )
        # associated_pairs = global_nearest_neighbor(
        #     self.segments + self.segment_nursery, observations, 
        #     self.orb_similarity, self.min_orb_matches / self.num_orb_features
        # )

        # separate segments associated with nursery and normal segments
        pairs_existing = [[seg_idx, obs_idx] for seg_idx, obs_idx \
                                in associated_pairs if seg_idx < len(self.segments)]
        pairs_nursery = [[seg_idx - len(self.segments), obs_idx] for seg_idx, obs_idx \
                                in associated_pairs if seg_idx >= len(self.segments)]

        # update segments with associated observations
        for seg_idx, obs_idx in pairs_existing:
            self.segments[seg_idx].update(observations[obs_idx], integrate_points=True)
        for seg_idx, obs_idx in pairs_nursery:
            # forcing add does not try to reconstruct the segment
            self.segment_nursery[seg_idx].update(observations[obs_idx], force=True, integrate_points=True)

        # delete masks for segments that were not seen recently
        for seg in self.segments:
            if not np.allclose(t, seg.last_seen, rtol=0.0):
                seg.last_observation.mask = None

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

        self.merge()
            
        return

    def mask_similarity(self, segment: Segment, observation: Observation, projected: bool = False):
        """
        Compute the similarity between the mask of a segment and an observation
        """
        if not projected or segment in self.segment_nursery:
            segment_last_mask = segment.last_observation.mask_downsampled
            if segment_last_mask is None:
                iou = 0.0
            else:
                iou = Tracker.compute_iou(segment_last_mask, observation.mask_downsampled)

        # compute the similarity using the projected mask rather than last mask
        else:
            segment_mask = segment.reconstruct_mask(observation.pose, downsample_factor=self.mask_downsample_factor)
            iou = Tracker.compute_iou(segment_mask, observation.mask_downsampled)
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
    
    def orb_similarity(self, segment: Segment, observation: Observation):
        """
        Compute the similarity between the ORB descriptors of a segment and an observation
        """
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        # find 2 best matches for each observation descriptor 
        # (will use to throw away ambiguous matches)
        matches = matcher.knnMatch(segment.last_observation.descriptors, 
                                   observation.descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < .75*n.distance:
                good_matches.append(m)

        return len(good_matches) / self.num_orb_features

    
    def filter_segment_graveyard(self, rm_fun: callable):
        """
        Filter the segment graveyard when rm_fun returns True
        """
        self.segment_graveyard = [seg for seg in self.segment_graveyard if not rm_fun(seg)]
        return
    
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
            extent = np.sort(seg.extent) # in ascending order
            if seg.num_points == 0:
                to_delete.append(seg)
                # reason.append(f"Segment {seg.id} has no points")
            elif seg.volume() < min_volume:
                to_delete.append(seg)
                # reason.append(f"Segment {seg.id} has volume {seg.volume} < {min_volume}")
            elif extent[-1] < min_max_extent:
                to_delete.append(seg)
                # reason.append(f"Segment {seg.id} has max extent {np.max(seg.extent)} < {min_max_extent}"
            elif extent[2] > plane_prune_params[0] and extent[1] > plane_prune_params[1] and extent[0] < plane_prune_params[2]:
                to_delete.append(seg)
                # reason.append(f"Segment {seg.id} has extent {seg.extent} which is likely a plane")
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

        max_time_passed_merge = 10.0
        max_iter = 100
        n = 0
        edited = True

        self.segment_graveyard = self.remove_bad_segments(self.segment_graveyard, min_max_extent=self.min_max_extent, plane_prune_params=self.plane_prune_params)
        self.segments = self.remove_bad_segments(self.segments)

        # repeatedly try to merge until no further merges are possible
        while n < max_iter and edited:
            edited = False
            n += 1

            for i, seg1 in enumerate(self.segments):
                for j, seg2 in enumerate(self.segments + self.segment_graveyard):
                    if i >= j:
                        continue

                    # TODO: merging needs to be faster, this may not be the right way to do it.
                    if np.abs(seg2.last_seen - seg1.last_seen) > max_time_passed_merge:
                        continue
                    combined_pts = np.vstack([seg1.points, seg2.points])
                    if combined_pts.shape[0] < 4:
                        iou3d = 0.0
                    else:
                        union_obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                                    o3d.utility.Vector3dVector(combined_pts))
                        intersection3d = seg1.volume() + seg2.volume() - union_obb.volume()
                        iou3d = intersection3d / union_obb.volume()

                    maks1 = seg1.reconstruct_mask(self.last_pose)
                    maks2 = seg2.reconstruct_mask(self.last_pose)
                    intersection2d = np.logical_and(maks1, maks2).sum()
                    union2d = np.logical_or(maks1, maks2).sum()
                    iou2d = intersection2d / union2d

                    if iou3d > self.merge_objects_iou_3d or iou2d > self.merge_objects_iou_2d:
                        for obs in seg2.observations:
                            # none of the observations will have masks, so need to update with 
                            # the last_observation copy (which will have a mask) instead
                            if obs.time == seg2.last_seen:
                                obs = seg2.last_observation
                            seg1.update(obs, integrate_points=False)
                        seg1.integrate_points_from_segment(seg2)
                        seg1.id = min(seg1.id, seg2.id)
                        if j < len(self.segments):
                            self.segments.pop(j)
                        else:
                            self.segment_graveyard.pop(j - len(self.segments))
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
            
    def make_pickle_compatible(self):
        """
        Make the tracker object pickle compatible
        """
        for seg in self.segments + self.segment_nursery + self.segment_graveyard:
            seg.reset_obb()
        return
            