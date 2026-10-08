import pickle
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from roman.map.map import ROMANMap
from roman.map.mapper import Mapper
from roman.params.mapper_params import MapperParams


def pose(x=0.0, angle=0.0):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('z', angle, degrees=True).as_matrix()
    result[0, 3] = x
    return result


def mapper(**kwargs):
    params = dict(store_aggregated_pcds=True, pcd_window_num_scans=2,
                  pcd_voxel_size_m=0.01)
    params.update(kwargs)
    return Mapper(MapperParams(**params), camera_params=None)


class MapperPointCloudTests(unittest.TestCase):
    def test_disabled_and_incomplete_windows(self):
        disabled = mapper(store_aggregated_pcds=False)
        disabled.update_point_cloud(0, pose(), np.array([[0, 0, 1]]))
        self.assertIsNone(disabled.get_roman_map().point_clouds)
        enabled = mapper()
        enabled.update_point_cloud(0, pose(), np.array([[0, 0, 1]]))
        self.assertEqual(enabled.get_roman_map().point_clouds, [])
        self.assertEqual(len(enabled._pcd_window), 1)

    def test_motion_gating_is_independent_of_segment_updates(self):
        subject = mapper(pcd_window_num_scans=3)
        scan = np.array([[0, 0, 1]])
        subject.update_point_cloud(0, pose(), scan)
        subject.update(0.5, pose(10), [], None)
        subject.update_point_cloud(1, pose(0.15), scan)
        subject.update_point_cloud(2, pose(0.25), scan)
        subject.update_point_cloud(3, pose(0.25, 30), scan)
        cloud = subject.get_roman_map().point_clouds[0]
        self.assertEqual((cloud.t0, cloud.tf), (0, 3))
        self.assertEqual(len(subject.times_history), 1)
        self.assertEqual(len(subject.aggregated_point_clouds), 1)

    def test_transform_filtering_and_input_ownership(self):
        subject = mapper(pcd_window_num_scans=1)
        scan = np.array([[1, 0, 1], [np.nan, 0, 1], [0, np.inf, 1],
                         [0, 0, 0], [0, 0, 6]])
        original = scan.copy()
        transform = pose(2, 90)
        subject.update_point_cloud(4, transform, scan)
        np.testing.assert_equal(scan, original)
        cloud = subject.get_roman_map().point_clouds[0]
        np.testing.assert_allclose(cloud.point_cloud, [[2, 1, 1]], atol=1e-12)
        self.assertEqual(cloud.frame, 'odom')
        self.assertEqual((cloud.t0, cloud.tf), (4, 4))
        scan[:] = 100
        transform[:] = 0
        np.testing.assert_allclose(cloud.point_cloud, [[2, 1, 1]], atol=1e-12)
        np.testing.assert_allclose(subject._last_pcd_pose, pose(2, 90))

    def test_empty_scans_do_not_advance_gate(self):
        subject = mapper()
        subject.update_point_cloud(0, pose(), np.empty((0, 3)))
        subject.update_point_cloud(1, pose(), np.array([[0, 0, 1]]))
        subject.update_point_cloud(2, pose(1), np.array([[np.nan, 0, 1]]))
        subject.update_point_cloud(3, pose(0.25), np.array([[0, 0, 1]]))
        self.assertEqual(len(subject.aggregated_point_clouds), 1)
        cloud = subject.aggregated_point_clouds[0]
        self.assertEqual((cloud.t0, cloud.tf), (1, 3))

    def test_overlap_and_zero_overlap(self):
        for overlap, expected_intervals, remaining in [
                (0.5, [(0, 3), (2, 5), (4, 7)], 2),
                (0.0, [(0, 3), (4, 7)], 0),
                (0.99, [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7)], 3)]:
            with self.subTest(overlap=overlap):
                subject = mapper(pcd_window_num_scans=4, pcd_window_overlap=overlap)
                for t in range(8):
                    subject.update_point_cloud(t, pose(t), np.array([[0, 0, 1]]))
                self.assertEqual([(cloud.t0, cloud.tf) for cloud in subject.aggregated_point_clouds],
                                 expected_intervals)
                self.assertEqual(len(subject._pcd_window), remaining)
                self.assertEqual(len(subject.aggregated_point_clouds[0].point_cloud), 4)

    def test_voxel_downsampling(self):
        subject = mapper(pcd_dist_thresh_m=0, pcd_voxel_size_m=0.1)
        subject.update_point_cloud(0, pose(), np.array([[0, 0, 1]]))
        subject.update_point_cloud(1, pose(), np.array([[0.01, 0, 1]]))
        np.testing.assert_allclose(subject.aggregated_point_clouds[0].point_cloud,
                                   [[0.005, 0, 1]])

    def test_map_export_and_pickle(self):
        subject = mapper(pcd_window_num_scans=1)
        subject.update_point_cloud(0, pose(), np.array([[0, 0, 1]]))
        exported = subject.get_roman_map()
        subject.update_point_cloud(1, pose(1), np.array([[0, 0, 1]]))
        self.assertEqual(len(exported.point_clouds), 1)
        restored = pickle.loads(pickle.dumps(exported))
        cloud = restored.point_clouds[0]
        self.assertEqual((cloud.t0, cloud.tf), (0, 0))
        np.testing.assert_equal(restored.point_clouds[0].point_cloud, [[0, 0, 1]])

    def test_concatenation_preserves_clouds_and_disabled_maps(self):
        subject = mapper(pcd_window_num_scans=1)
        subject.update_point_cloud(0, pose(), np.array([[0, 0, 1]]))
        clouds = subject.get_roman_map().point_clouds
        for first, second, expected in [(clouds, clouds, 2), (None, clouds, 1),
                                        (clouds, None, 1), (None, None, None)]:
            maps = [ROMANMap([SimpleNamespace(id=0)], [pose()], [t],
                             point_clouds=value)
                    for t, value in enumerate((first, second))]
            combined = ROMANMap.concatenate(maps)
            if expected is None:
                self.assertIsNone(combined.point_clouds)
            else:
                self.assertEqual(len(combined.point_clouds), expected)

    def test_parameter_validation(self):
        invalid = dict(pcd_window_num_scans=[0, -1, 1.5, True],
                       pcd_window_overlap=[-0.1, 1, np.nan],
                       pcd_dist_thresh_m=[-1, np.inf],
                       pcd_ang_thresh_deg=[-1, np.nan],
                       pcd_voxel_size_m=[0, -1, np.inf],
                       pcd_max_depth=[0, -1, np.nan])
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    mapper(**{name: value})

    def test_concatenation_of_point_cloud_only_maps(self):
        subject = mapper(pcd_window_num_scans=1)
        subject.update_point_cloud(0, pose(), np.array([[0, 0, 1]]))
        exported = subject.get_roman_map()
        combined = ROMANMap.concatenate([exported, exported])
        self.assertEqual(len(combined.point_clouds), 2)
        self.assertEqual(combined.trajectory, [])


if __name__ == '__main__':
    unittest.main()
