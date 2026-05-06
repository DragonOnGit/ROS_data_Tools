import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from occupancy_grid import (OccupancyGridMap, OccupancyGridConfig,
                            CoordinateTransformer, TrajectoryOverlay)
from octree_map import OctreeMap, OctreeConfig


class TestCoordinateTransformer(unittest.TestCase):
    def setUp(self):
        self.ct = CoordinateTransformer(flip_x=True, flip_y=True)
        self.ct_no_flip = CoordinateTransformer(flip_x=False, flip_y=False)
        self.ct_offset = CoordinateTransformer(flip_x=True, flip_y=True, offset_x=10.0, offset_y=20.0)

    def test_transform_point_basic(self):
        p = np.array([3.0, 4.0])
        tp = self.ct.transform_point(p)
        np.testing.assert_array_almost_equal(tp, [-3.0, -4.0])

    def test_transform_point_3d(self):
        p = np.array([3.0, 4.0, 5.0])
        tp = self.ct.transform_point(p)
        np.testing.assert_array_almost_equal(tp, [-3.0, -4.0, 5.0])

    def test_transform_point_no_flip(self):
        p = np.array([3.0, 4.0])
        tp = self.ct_no_flip.transform_point(p)
        np.testing.assert_array_almost_equal(tp, [3.0, 4.0])

    def test_transform_point_with_offset(self):
        p = np.array([3.0, 4.0])
        tp = self.ct_offset.transform_point(p)
        np.testing.assert_array_almost_equal(tp, [7.0, 16.0])

    def test_transform_trajectory(self):
        traj = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ttraj = self.ct.transform_trajectory(traj)
        expected = np.array([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]])
        np.testing.assert_array_almost_equal(ttraj, expected)

    def test_round_trip_precision(self):
        pts = np.random.randn(100, 2) * 10
        transformed = self.ct.transform_trajectory(pts)
        back = self.ct.transform_trajectory(transformed)
        np.testing.assert_array_almost_equal(pts, back, decimal=10)


class TestTrajectoryOverlay(unittest.TestCase):
    def test_default_values(self):
        overlay = TrajectoryOverlay()
        self.assertTrue(overlay.show_all)
        self.assertTrue(overlay.show_actual)
        self.assertTrue(overlay.show_expected)
        self.assertEqual(overlay.actual_color, (0, 0, 1.0))
        self.assertEqual(overlay.expected_color, (1.0, 0, 0))
        self.assertEqual(overlay.arrow_interval, 5.0)
        self.assertEqual(overlay.arrow_length, 6.0)

    def test_show_all_false(self):
        overlay = TrajectoryOverlay(show_all=False)
        self.assertFalse(overlay.show_all)

    def test_custom_colors(self):
        overlay = TrajectoryOverlay(actual_color=(0, 1, 0), expected_color=(1, 1, 0))
        self.assertEqual(overlay.actual_color, (0, 1, 0))
        self.assertEqual(overlay.expected_color, (1, 1, 0))


class TestOccupancyGridExtract(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        sensor = np.array([0.0, 0.0])
        n_pts = 1000
        angles = np.random.uniform(0, 2*np.pi, n_pts)
        dists = np.random.uniform(2.0, 10.0, n_pts)
        self.pts = np.column_stack([
            sensor[0] + dists * np.cos(angles),
            sensor[1] + dists * np.sin(angles),
            np.random.uniform(0, 3, n_pts)
        ])
        config_g = OccupancyGridConfig(resolution=0.1, use_raycast=True, ray_max_dist=15.0)
        self.ogm_global = OccupancyGridMap(config_g)
        self.ogm_global.build(self.pts, sensor_positions=sensor.reshape(1, -1))

    def test_extract_at_center(self):
        config_l = OccupancyGridConfig(resolution=0.1, local_range=10.0)
        ogm_local = OccupancyGridMap(config_l)
        center = np.array([0.0, 0.0])
        result = ogm_local.extract_local_from_global(self.ogm_global, center, local_range=10.0)
        self.assertTrue(result)
        self.assertIsNotNone(ogm_local.grid)
        stats = ogm_local.get_statistics()
        self.assertGreater(stats['total_cells'], 0)

    def test_extract_at_offset(self):
        config_l = OccupancyGridConfig(resolution=0.1, local_range=8.0)
        ogm_local = OccupancyGridMap(config_l)
        center = np.array([5.0, 5.0])
        result = ogm_local.extract_local_from_global(self.ogm_global, center, local_range=8.0)
        self.assertTrue(result)
        bounds = ogm_local.local_bounds
        self.assertIsNotNone(bounds)

    def test_extract_out_of_range(self):
        config_l = OccupancyGridConfig(resolution=0.1, local_range=10.0)
        ogm_local = OccupancyGridMap(config_l)
        center = np.array([100.0, 100.0])
        result = ogm_local.extract_local_from_global(self.ogm_global, center, local_range=10.0)
        self.assertTrue(result)
        stats = ogm_local.get_statistics()
        self.assertGreater(stats['unknown_cells'], 0)

    def test_local_bounds(self):
        config_l = OccupancyGridConfig(resolution=0.1, local_range=10.0)
        ogm_local = OccupancyGridMap(config_l)
        center = np.array([0.0, 0.0])
        ogm_local.extract_local_from_global(self.ogm_global, center, local_range=10.0)
        bounds = ogm_local.local_bounds
        min_xy, max_xy = bounds
        self.assertAlmostEqual(min_xy[0], -5.0, delta=0.2)
        self.assertAlmostEqual(max_xy[0], 5.0, delta=0.2)


class TestVisibleSegments(unittest.TestCase):
    def setUp(self):
        config = OccupancyGridConfig(resolution=0.1, local_range=10.0)
        self.ogm = OccupancyGridMap(config)

    def test_single_segment(self):
        traj = np.array([[-20, 0], [-10, 0], [0, 0], [1, 0], [2, 0], [3, 0], [20, 0], [30, 0]])
        segments = self.ogm._extract_visible_segments(traj, -5, 5, -5, 5)
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), 4)

    def test_two_segments(self):
        traj = np.array([[0, 0], [1, 0], [20, 0], [21, 0], [2, 0], [3, 0]])
        segments = self.ogm._extract_visible_segments(traj, -5, 5, -5, 5)
        self.assertEqual(len(segments), 2)

    def test_all_visible(self):
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        segments = self.ogm._extract_visible_segments(traj, -5, 5, -5, 5)
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), 4)

    def test_none_visible(self):
        traj = np.array([[20, 0], [30, 0], [40, 0]])
        segments = self.ogm._extract_visible_segments(traj, -5, 5, -5, 5)
        self.assertEqual(len(segments), 0)


class TestDisplayModeLogic(unittest.TestCase):
    def test_mode_mapping(self):
        mode_map = {0: 'filtered', 1: 'raw', 2: 'both'}
        self.assertEqual(mode_map[0], 'filtered')
        self.assertEqual(mode_map[1], 'raw')
        self.assertEqual(mode_map[2], 'both')

    def test_mode_in_checks(self):
        mode = 'both'
        self.assertIn('filtered', ('filtered', 'both'))
        self.assertIn('raw', ('raw', 'both'))
        mode = 'filtered'
        self.assertIn('filtered', ('filtered', 'both'))
        self.assertNotIn('raw', ('filtered',))


class TestPipeVisualization(unittest.TestCase):
    def test_pipe_data_structure(self):
        from bag_parser import PoseData
        ts = np.array([0.0, 1.0, 2.0])
        upper = PoseData(timestamp=ts, x=np.array([0, 1, 2]), y=np.array([0, 0, 0]),
                        z=np.array([3, 3, 3]), roll=np.zeros(3), pitch=np.zeros(3), yaw=np.zeros(3))
        lower = PoseData(timestamp=ts, x=np.array([0, 1, 2]), y=np.array([0, 0, 0]),
                        z=np.array([1, 1, 1]), roll=np.zeros(3), pitch=np.zeros(3), yaw=np.zeros(3))
        self.assertEqual(len(upper.x), 3)
        self.assertEqual(len(lower.x), 3)
        self.assertTrue(np.all(upper.z > lower.z))

    def test_pipe_vertices_calculation(self):
        n = 10
        ux = np.linspace(0, 10, n)
        uy = np.zeros(n)
        uz = np.full(n, 3.0)
        lx = np.linspace(0, 10, n)
        ly = np.zeros(n)
        lz = np.full(n, 1.0)
        step = max(1, n // 5)
        verts_count = 0
        for i in range(0, n - step, step):
            j = min(i + step, n - 1)
            verts = [
                [ux[i], uy[i], uz[i]],
                [ux[j], uy[j], uz[j]],
                [lx[j], ly[j], lz[j]],
                [lx[i], ly[i], lz[i]],
            ]
            self.assertEqual(len(verts), 4)
            verts_count += 1
        self.assertGreater(verts_count, 0)


if __name__ == '__main__':
    unittest.main()
