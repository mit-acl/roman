import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from roman.map.mapper import Mapper
from roman.params.mapper_params import MapperParams
from roman.utils import DepthProjectionMetadata, unproject_depth


class DepthProjectionTests(unittest.TestCase):
    def metadata(self, scale=1000., stride=1):
        return DepthProjectionMetadata(stride, scale,
            o3d.camera.PinholeCameraIntrinsic(3, 3, 2., 2., 1., 1.))

    def test_units_stride_and_invalid_depth(self):
        depth = np.array([[1000., 0., 2000.], [np.nan, np.inf, -1.], [3000., 1000000., 4000.]])
        original = depth.copy()
        metadata = self.metadata(stride=2)
        xyz, valid = unproject_depth(depth, metadata.pcd_stride, metadata.depth_scale, metadata.depth_intrinsics)
        np.testing.assert_allclose(xyz, [[[-.5, -.5, 1.], [1., -1., 2.]],
                                         [[-1.5, 1.5, 3.], [2., 2., 4.]]])
        self.assertTrue(valid.all())
        _, valid = unproject_depth(depth, 1, 1000., metadata.depth_intrinsics)
        np.testing.assert_equal(valid, [[True, False, True], [False, False, False], [True, False, True]])
        meters, _ = unproject_depth(depth / 1000., 2, 1., metadata.depth_intrinsics)
        np.testing.assert_equal(meters, xyz)
        np.testing.assert_equal(depth, original)

    def test_projection_matches_open3d(self):
        depth = np.array([[1000, 2000, 3000], [4000, 0, 1000], [2000, 3000, 4000]], dtype=np.uint16)
        metadata = self.metadata(stride=2)
        xyz, valid = unproject_depth(depth, 2, 1000., metadata.depth_intrinsics)
        expected = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth), metadata.depth_intrinsics,
            depth_scale=1000., depth_trunc=1000., stride=2)
        np.testing.assert_allclose(xyz[valid], np.asarray(expected.points))

    def test_mapper_gates_before_unprojection_and_keeps_pointcloud_support(self):
        mapper = Mapper(MapperParams(store_aggregated_pcds=True, pcd_window_num_scans=1,
            pcd_window_overlap=0., pcd_voxel_size_m=.01), camera_params=None)
        depth = np.full((3, 3), 1000, dtype=np.uint16)
        metadata = self.metadata()
        with patch('roman.map.mapper.unproject_depth', wraps=unproject_depth) as project:
            mapper.update_point_cloud(0., np.eye(4), depth, is_depth_img=True,
                                      depth_projection_metadata=metadata)
            mapper.update_point_cloud(1., np.eye(4), depth, is_depth_img=True,
                                      depth_projection_metadata=metadata)
            self.assertEqual(project.call_count, 1)
            moved = np.eye(4)
            moved[0, 3] = 1.
            mapper.update_point_cloud(2., moved, depth, is_depth_img=True,
                                      depth_projection_metadata=metadata)
            self.assertEqual(project.call_count, 2)
            moved[0, 3] = 2.
            mapper.update_point_cloud(3., moved, np.array([[0., 0., 1.]]))
            self.assertEqual(project.call_count, 2)
        self.assertEqual(len(mapper.aggregated_point_clouds), 3)
        np.testing.assert_allclose(mapper.aggregated_point_clouds[-1].point_cloud, [[2., 0., 1.]])

    def test_missing_metadata_and_invalid_projection_parameters(self):
        mapper = Mapper(MapperParams(store_aggregated_pcds=True), camera_params=None)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            mapper.update_point_cloud(0., np.eye(4), np.ones((3, 3)), is_depth_img=True)
        for scale, stride in [(0., 1), (np.inf, 1), (1000., 0), (1000., 1.5)]:
            with self.assertRaises(ValueError):
                self.metadata(scale, stride)
