import unittest
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from roman.map.map import AggregatedPointCloud, ROMANMap, SubmapParams, submaps_from_roman_map


@dataclass
class TestSegment:
    id: int
    center: np.ndarray
    first_seen: float
    last_seen: float

    def set_center_ref(self, reference):
        pass

    def minimal_data(self):
        return self

    def reference_time(self):
        return (self.first_seen + self.last_seen) / 2

    def transform(self, matrix):
        self.center = matrix[:3, :3] @ self.center + matrix[:3, 3]


def cloud(t0, tf, x, frame='odom'):
    return AggregatedPointCloud(t0, tf, np.array([[x, 0., 1.]]), frame)


def make_map(clouds, segments=None, poses=None, times=None):
    return ROMANMap(
        segments=segments if segments is not None else [TestSegment(0, np.zeros(3), 10, 20)],
        trajectory=poses if poses is not None else [np.eye(4)],
        times=np.array(times if times is not None else [15.]),
        point_clouds=clouds,
    )


def params(force_fill=False, minimal=True, **kwargs):
    values = dict(force_fill_submaps=force_fill, use_minimal_data=minimal,
                  max_size=2, overlap=0, radius=None, include_point_cloud=True)
    values.update(kwargs)
    return SubmapParams(**values)


class SubmapPointCloudTests(unittest.TestCase):
    def test_inclusive_interval_overlap_in_both_submap_modes(self):
        clouds = [cloud(0, 9, 0), cloud(5, 10, 1), cloud(8, 12, 2),
                  cloud(12, 18, 3), cloud(0, 30, 4), cloud(18, 22, 5),
                  cloud(20, 25, 6), cloud(21, 30, 7)]
        expected = np.array([[x, 0, 1] for x in range(1, 7)])
        for force_fill in (False, True):
            for minimal in (False, True):
                with self.subTest(force_fill=force_fill, minimal=minimal):
                    submaps = submaps_from_roman_map(make_map(clouds), params(force_fill, minimal))
                    points = submaps[0].point_cloud
                    np.testing.assert_equal(points[np.argsort(points[:, 0])], expected)

    def test_clouds_are_transformed_to_submap_frame_without_mutating_sources(self):
        center_pose = np.eye(4)
        center_pose[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        center_pose[:3, 3] = [2, 3, 0]
        aggregate = AggregatedPointCloud(10, 20, np.array([[2., 4., 1.]]))
        source = aggregate.point_cloud.copy()
        for force_fill in (False, True):
            with self.subTest(force_fill=force_fill):
                submaps = submaps_from_roman_map(
                    make_map([aggregate], poses=[center_pose.copy()]), params(force_fill))
                np.testing.assert_allclose(submaps[0].point_cloud, [[1, 0, 1]], atol=1e-12)
                np.testing.assert_equal(aggregate.point_cloud, source)

    def test_time_span_uses_segments_after_pruning(self):
        segments = [TestSegment(0, np.zeros(3), 10, 20),
                    TestSegment(1, np.zeros(3), 0, 100)]
        submaps = submaps_from_roman_map(
            make_map([cloud(12, 18, 1), cloud(80, 90, 2)], segments), params(max_size=1))
        self.assertEqual(len(submaps[0].segments), 1)
        np.testing.assert_equal(submaps[0].point_cloud, [[1, 0, 1]])

    def test_absent_empty_and_nonoverlapping_clouds(self):
        for clouds in (None, [], [cloud(0, 9, 1)],
                       [AggregatedPointCloud(10, 20, np.empty((0, 3)))]):
            with self.subTest(clouds=clouds):
                submaps = submaps_from_roman_map(make_map(clouds), params())
                if clouds is None:
                    self.assertIsNone(submaps[0].point_cloud)
                else:
                    self.assertEqual(submaps[0].point_cloud.shape, (0, 3))

    def test_point_cloud_inclusion_can_be_disabled(self):
        submaps = submaps_from_roman_map(make_map([cloud(10, 20, 1)]),
                                        params(include_point_cloud=False))
        self.assertIsNone(submaps[0].point_cloud)

    def test_downsampling_combines_repeated_and_nearby_points(self):
        clouds = [cloud(10, 20, 1), cloud(10, 20, 1), cloud(10, 20, 1.01),
                  cloud(10, 20, 2)]
        sources = [aggregate.point_cloud.copy() for aggregate in clouds]
        for force_fill in (False, True):
            with self.subTest(force_fill=force_fill):
                submaps = submaps_from_roman_map(make_map(clouds), params(force_fill))
                points = submaps[0].point_cloud
                np.testing.assert_allclose(points[np.argsort(points[:, 0])],
                                           [[1 + 0.01 / 3, 0, 1], [2, 0, 1]])
                for aggregate, source in zip(clouds, sources):
                    np.testing.assert_equal(aggregate.point_cloud, source)
        submaps = submaps_from_roman_map(make_map(clouds), params(point_cloud_voxel_size=0.001))
        self.assertEqual(len(submaps[0].point_cloud), 3)

    def test_invalid_voxel_sizes(self):
        for voxel_size in (0, -0.1, np.nan, np.inf):
            with self.subTest(voxel_size=voxel_size), self.assertRaises(ValueError):
                submaps_from_roman_map(make_map([cloud(10, 20, 1)]),
                                      params(point_cloud_voxel_size=voxel_size))

    def test_each_submap_uses_its_own_segments_time_span(self):
        segments = [TestSegment(0, np.array([0., 0., 0.]), 0, 2),
                    TestSegment(1, np.array([20., 0., 0.]), 10, 12)]
        other_pose = np.eye(4)
        other_pose[0, 3] = 20
        submaps = submaps_from_roman_map(
            make_map([cloud(0, 2, 1), cloud(10, 12, 21)], segments,
                     poses=[np.eye(4), other_pose], times=[1, 11]), params(radius=5))
        self.assertEqual(len(submaps), 2)
        for submap in submaps:
            np.testing.assert_allclose(submap.point_cloud, [[1, 0, 1]])

    def test_no_segments_and_unexpected_aggregate_frame(self):
        self.assertEqual(submaps_from_roman_map(make_map([cloud(10, 20, 1)], []), params()), [])
        with self.assertRaises(ValueError):
            submaps_from_roman_map(make_map([cloud(10, 20, 1, frame='camera')]), params())


if __name__ == '__main__':
    unittest.main()
