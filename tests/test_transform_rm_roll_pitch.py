import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from roman.map.map import Submap
from roman.utils import transform_rm_roll_pitch


class TransformRemoveRollPitchTests(unittest.TestCase):
    def test_returns_independent_transform_and_preserves_input(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('ZYX', [.7, -.3, .2]).as_matrix()
        pose[:3, 3] = [1., 2., 3.]
        original = pose.copy()
        result = transform_rm_roll_pitch(pose)
        np.testing.assert_array_equal(pose, original)
        np.testing.assert_allclose(result[:3, :3], Rotation.from_euler('z', .7).as_matrix())
        np.testing.assert_array_equal(result[:3, 3], pose[:3, 3])
        np.testing.assert_array_equal(result[3], pose[3])
        self.assertFalse(np.shares_memory(result, pose))
        result[0, 3] = 10.
        np.testing.assert_array_equal(pose, original)

    def test_submap_properties_preserve_estimated_and_ground_truth_poses(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('xyz', [.2, -.3, .7]).as_matrix()
        gt = pose.copy()
        submap = Submap(0, 0., [], pose, pose_flu_gt=gt)
        original = pose.copy()
        for aligned in [submap.pose_gravity_aligned, submap.pose_gravity_aligned_gt]:
            np.testing.assert_allclose(aligned, transform_rm_roll_pitch(original))
        np.testing.assert_array_equal(pose, original)
        np.testing.assert_array_equal(gt, original)
