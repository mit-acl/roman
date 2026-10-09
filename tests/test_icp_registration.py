import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from roman.align.object_registration import ObjectRegistration, InsufficientAssociationsException
from roman.map.map import SubmapParams, ROMANMap, AggregatedPointCloud
from roman.object.object import Object
from roman.params.submap_align_params import SubmapAlignParams, SubmapAlignInputOutput


class SubmapObject(Object):
    first_seen = 0
    last_seen = 1

    def set_center_ref(self, reference):
        pass

    def minimal_data(self):
        return self


class ICPRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.source = np.random.default_rng(7).uniform(-3, 3, (120, 3))
        self.T = np.eye(4)
        self.T[:3, :3] = Rotation.from_euler('xyz', [0.03, -0.02, 0.15]).as_matrix()
        self.T[:3, 3] = [0.5, -0.3, 0.1]
        self.target = self.source @ self.T[:3, :3].T + self.T[:3, 3]
        self.map1 = [Object(point + [0.06, 0.02, -0.03]) for point in self.target[:6]]
        self.map2 = [Object(point) for point in self.source[:6]]
        self.associations = np.column_stack((np.arange(6), np.arange(6)))
        self.registration = SubmapAlignParams(icp_on_submap_pcds=True).get_object_registration()

    def test_object_guess_is_refined_in_map2_to_map1_direction(self):
        initial = ObjectRegistration().T_align(self.map1, self.map2, self.associations)
        self.assertGreater(np.linalg.norm(initial - self.T), 0.01)
        target, source = self.target.copy(), self.source.copy()
        actual = self.registration.refine_with_icp(initial, target, source)
        np.testing.assert_allclose(actual, self.T, atol=1e-10)
        np.testing.assert_equal(target, self.target)
        np.testing.assert_equal(source, self.source)

    def test_object_alignment_does_not_run_icp(self):
        with patch.object(self.registration, 'refine_with_icp', side_effect=AssertionError):
            actual = self.registration.T_align(self.map1, self.map2, self.associations)
        expected = ObjectRegistration().T_align(self.map1, self.map2, self.associations)
        np.testing.assert_equal(actual, expected)

    def test_yaw_only_refinement_preserves_gravity_constraint(self):
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler('z', 0.15).as_matrix()
        expected[:3, 3] = self.T[:3, 3]
        target = self.source @ expected[:3, :3].T + expected[:3, 3]
        map1 = [Object(point + [0.06, 0.02, -0.03]) for point in target[:6]]
        initial = self.registration.T_align(map1, self.map2, self.associations, xyz_yaw_only=True)
        actual = self.registration.refine_with_icp(initial, target, self.source, xyz_yaw_only=True)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        np.testing.assert_allclose(actual[:3, 2], [0, 0, 1], atol=1e-12)

    def test_missing_empty_nonoverlapping_or_degenerate_clouds_keep_initial(self):
        initial = self.T.copy()
        line = np.column_stack((np.arange(10), np.zeros((10, 2))))
        for target, source in [(None, self.source), (self.target, None),
                               (np.empty((0, 3)), self.source),
                               (self.target[:2], self.source),
                               (self.target + 100, self.source), (line, line)]:
            with self.subTest(target=target, source=source):
                guess = np.eye(4) if target is line else initial
                actual = self.registration.refine_with_icp(guess, target, source)
                np.testing.assert_equal(actual, guess)

    def test_low_fitness_keeps_object_guess(self):
        initial = self.T.copy()
        initial[:3, 3] += [0.06, 0.02, -0.03]
        actual = self.registration.refine_with_icp(initial, self.target[:6], self.source)
        np.testing.assert_equal(actual, initial)

    def test_worse_fit_keeps_object_guess(self):
        correction = np.eye(4)
        correction[0, 3] = 0.1
        self.registration.icp_max_iterations = 1
        with patch.object(self.registration, '_fit_transform', return_value=correction):
            actual = self.registration.refine_with_icp(self.T, self.target, self.source)
        np.testing.assert_equal(actual, self.T)

    def test_nonfinite_points_are_filtered(self):
        bad = np.array([[np.nan, 0, 0], [np.inf, 0, 0]])
        initial = self.registration.T_align(self.map1, self.map2, self.associations)
        actual = self.registration.refine_with_icp(
            initial, np.vstack((self.target, bad)), np.vstack((self.source, bad)))
        np.testing.assert_allclose(actual, self.T, atol=1e-10)

    def test_insufficient_object_associations_do_not_run_icp(self):
        with patch.object(self.registration, 'refine_with_icp', side_effect=AssertionError):
            with self.assertRaises(InsufficientAssociationsException):
                self.registration.T_align(self.map1, self.map2, self.associations[:2])

    def test_parameter_wiring_and_validation(self):
        params = SubmapAlignParams(icp_on_submap_pcds=True, icp_max_iterations=12,
                                   icp_max_correspondence_distance=0.4, icp_min_fitness=0.6)
        registration = params.get_object_registration()
        self.assertEqual(registration.icp_max_iterations, 12)
        self.assertEqual(registration.icp_max_correspondence_distance, 0.4)
        self.assertEqual(registration.icp_min_fitness, 0.6)
        self.assertTrue(SubmapParams.from_submap_align_params(params).include_point_cloud)
        self.assertFalse(SubmapParams.from_submap_align_params(SubmapAlignParams()).include_point_cloud)
        for invalid in ({'dim': 2}, {'icp_max_iterations': 0}, {'icp_max_iterations': 1.5},
                        {'icp_max_correspondence_distance': 0},
                        {'icp_max_correspondence_distance': np.inf},
                        {'icp_min_fitness': -0.1}, {'icp_min_fitness': np.nan}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                SubmapAlignParams(icp_on_submap_pcds=True, **invalid)

    def test_offline_alignment_passes_submap_clouds_to_refinement(self):
        from roman.align.submap_align import submap_align

        for yaw_only, icp_enabled in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(yaw_only=yaw_only, icp_enabled=icp_enabled):
                expected = self.T.copy()
                if yaw_only:
                    expected[:3, :3] = Rotation.from_euler('z', 0.15).as_matrix()
                target = self.source @ expected[:3, :3].T + expected[:3, 3]
                maps = [ROMANMap(
                    segments=[SubmapObject(point + bias) for point in points[:6]],
                    trajectory=[np.eye(4)], times=np.array([0.]),
                    point_clouds=[AggregatedPointCloud(0, 1, points)],
                ) for points, bias in [(target, np.array([0.06, 0.02, -0.03])),
                                        (self.source, np.zeros(3))]]
                params = SubmapAlignParams(icp_on_submap_pcds=icp_enabled,
                                          force_rm_lc_roll_pitch=yaw_only)
                io = SubmapAlignInputOutput(inputs=['map1', 'map2'], output_dir='', run_name='test')
                with patch('roman.align.submap_align.load_roman_map', side_effect=maps), \
                        patch('roman.align.submap_align.save_submap_align_results') as save, \
                        patch.object(self.registration, 'register', return_value=self.associations), \
                        patch.object(self.registration, 'refine_with_icp',
                                     wraps=self.registration.refine_with_icp) as refine, \
                        patch.object(SubmapAlignParams, 'get_object_registration', return_value=self.registration):
                    submap_align(params, io)
                if icp_enabled:
                    refine.assert_called_once()
                else:
                    refine.assert_not_called()
                    expected[:3, 3] += [0.06, 0.02, -0.03]
                results = save.call_args.args[0]
                np.testing.assert_allclose(results.T_ij_hat_mat[0, 0], expected, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
