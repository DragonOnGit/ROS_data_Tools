# -*- coding: utf-8 -*-
"""
八叉树地图模块单元测试
测试覆盖率目标: >= 80%
"""

import unittest
import numpy as np
import os
import tempfile

from octree_map import OctreeMap, OctreeConfig, OctreeNode
from octree_visualizer import OctreeVisualizer


def _generate_random_points(n: int = 1000, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, 3) * 5.0


def _generate_grid_points(resolution: int = 5) -> np.ndarray:
    x = np.linspace(-2, 2, resolution)
    y = np.linspace(-2, 2, resolution)
    z = np.linspace(0, 3, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


class TestOctreeConfig(unittest.TestCase):
    
    def test_default_config(self):
        config = OctreeConfig()
        self.assertEqual(config.voxel_size, 0.1)
        self.assertEqual(config.max_depth, 8)
        self.assertEqual(config.min_depth, 0)
        self.assertEqual(config.density_threshold, 5)
        self.assertEqual(config.color_scheme, 'height')
        self.assertEqual(config.transparency, 0.8)
        self.assertFalse(config.show_borders)
        self.assertEqual(config.render_quality, 'medium')
    
    def test_config_clamping(self):
        config = OctreeConfig(voxel_size=0.001, max_depth=20, min_depth=-1,
                             density_threshold=0, transparency=2.0)
        self.assertAlmostEqual(config.voxel_size, 0.01)
        self.assertEqual(config.max_depth, 16)
        self.assertEqual(config.min_depth, 0)
        self.assertEqual(config.density_threshold, 1)
        self.assertAlmostEqual(config.transparency, 1.0)
    
    def test_invalid_color_scheme(self):
        config = OctreeConfig(color_scheme='invalid')
        self.assertEqual(config.color_scheme, 'height')
    
    def test_invalid_render_quality(self):
        config = OctreeConfig(render_quality='ultra')
        self.assertEqual(config.render_quality, 'medium')
    
    def test_min_depth_greater_than_max(self):
        config = OctreeConfig(min_depth=10, max_depth=5)
        self.assertLessEqual(config.min_depth, config.max_depth)


class TestOctreeNode(unittest.TestCase):
    
    def test_default_node(self):
        node = OctreeNode()
        np.testing.assert_array_equal(node.center, np.zeros(3))
        self.assertEqual(node.size, 1.0)
        self.assertEqual(node.depth, 0)
        self.assertEqual(node.point_count, 0)
        self.assertTrue(node.is_leaf)
    
    def test_custom_node(self):
        center = np.array([1.0, 2.0, 3.0])
        node = OctreeNode(center=center, size=2.0, depth=3, point_count=10)
        np.testing.assert_array_equal(node.center, center)
        self.assertEqual(node.size, 2.0)
        self.assertEqual(node.depth, 3)
        self.assertEqual(node.point_count, 10)


class TestOctreeMap(unittest.TestCase):
    
    def setUp(self):
        self.points = _generate_random_points(500)
        self.config = OctreeConfig(voxel_size=0.5, max_depth=6, density_threshold=3)
        self.octree = OctreeMap(self.config)
    
    def test_build_basic(self):
        self.octree.build(self.points)
        self.assertIsNotNone(self.octree.root)
        self.assertGreater(self.octree.total_nodes, 1)
        self.assertGreater(self.octree.total_leaves, 0)
        self.assertGreater(self.octree.build_time, 0)
    
    def test_build_empty_points(self):
        with self.assertRaises(ValueError):
            self.octree.build(np.array([]).reshape(0, 3))
    
    def test_build_none_points(self):
        with self.assertRaises(ValueError):
            self.octree.build(None)
    
    def test_build_invalid_shape(self):
        with self.assertRaises(ValueError):
            self.octree.build(np.array([1, 2, 3]))
    
    def test_get_leaf_voxels(self):
        self.octree.build(self.points)
        voxels = self.octree.get_leaf_voxels()
        self.assertIsInstance(voxels, list)
        self.assertGreater(len(voxels), 0)
        for v in voxels:
            self.assertIn('center', v)
            self.assertIn('size', v)
            self.assertIn('depth', v)
            self.assertIn('point_count', v)
            self.assertIn('z', v)
            self.assertIn('density', v)
    
    def test_get_voxel_centers(self):
        self.octree.build(self.points)
        centers = self.octree.get_voxel_centers()
        self.assertEqual(centers.ndim, 2)
        self.assertEqual(centers.shape[1], 3)
    
    def test_get_voxel_sizes(self):
        self.octree.build(self.points)
        sizes = self.octree.get_voxel_sizes()
        self.assertEqual(sizes.ndim, 1)
        self.assertGreater(len(sizes), 0)
    
    def test_get_statistics(self):
        self.octree.build(self.points)
        stats = self.octree.get_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_points', stats)
        self.assertIn('total_leaves', stats)
        self.assertIn('build_time', stats)
        self.assertIn('voxel_size_min', stats)
        self.assertIn('density_mean', stats)
    
    def test_query_point(self):
        self.octree.build(self.points)
        voxels = self.octree.get_leaf_voxels()
        if voxels:
            query = voxels[0]['center']
            result = self.octree.query_point(query)
            self.assertIsNotNone(result)
            self.assertIn('center', result)
            self.assertIn('size', result)
            self.assertIn('point_count', result)
    
    def test_query_point_boundary(self):
        self.octree.build(self.points)
        result = self.octree.query_point(np.array([999.0, 999.0, 999.0]))
        self.assertIsNone(result)
    
    def test_query_point_no_tree(self):
        result = self.octree.query_point(np.zeros(3))
        self.assertIsNone(result)
    
    def test_ray_cast(self):
        self.octree.build(self.points)
        origin = np.array([-10.0, -10.0, -10.0])
        direction = np.array([1.0, 1.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        result = self.octree.ray_cast(origin, direction)
        # May or may not hit depending on point distribution
    
    def test_ray_cast_no_tree(self):
        result = self.octree.ray_cast(np.zeros(3), np.ones(3))
        self.assertIsNone(result)
    
    def test_height_filter(self):
        config = OctreeConfig(voxel_size=0.5, max_depth=6,
                             height_min=0.0, height_max=2.0)
        octree = OctreeMap(config)
        points = _generate_random_points(500)
        octree.build(points)
        
        voxels = octree.get_leaf_voxels()
        for v in voxels:
            self.assertGreaterEqual(v['z'], 0.0 - v['size'])
            self.assertLessEqual(v['z'], 2.0 + v['size'])
    
    def test_export_voxels(self):
        self.octree.build(self.points)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name
        try:
            result = self.octree.export_voxels(filepath)
            self.assertTrue(result)
            self.assertTrue(os.path.exists(filepath))
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_export_no_data(self):
        result = self.octree.export_voxels('/tmp/test.npz')
        self.assertFalse(result)
    
    def test_grid_points(self):
        points = _generate_grid_points(5)
        config = OctreeConfig(voxel_size=0.3, max_depth=5)
        octree = OctreeMap(config)
        octree.build(points)
        self.assertGreater(octree.total_leaves, 0)
    
    def test_large_depth(self):
        config = OctreeConfig(voxel_size=0.01, max_depth=12, density_threshold=1)
        octree = OctreeMap(config)
        points = _generate_random_points(100)
        octree.build(points)
        self.assertGreater(octree.total_leaves, 0)
    
    def test_min_depth(self):
        config = OctreeConfig(voxel_size=0.5, max_depth=6, min_depth=2)
        octree = OctreeMap(config)
        octree.build(self.points)
        voxels = octree.get_leaf_voxels()
        depths = [v['depth'] for v in voxels]
        self.assertGreaterEqual(min(depths), 2)
    
    def test_empty_after_filter(self):
        config = OctreeConfig(height_min=1000.0, height_max=2000.0)
        octree = OctreeMap(config)
        octree.build(self.points)
        voxels = octree.get_leaf_voxels()
        self.assertEqual(len(voxels), 0)


class TestOctreeVisualizer(unittest.TestCase):
    
    def setUp(self):
        self.points = _generate_random_points(200)
        self.config = OctreeConfig(voxel_size=0.5, max_depth=4, density_threshold=3)
        self.octree = OctreeMap(self.config)
        self.octree.build(self.points)
        self.visualizer = OctreeVisualizer(self.octree)
    
    def test_visualize_3d(self):
        import matplotlib
        matplotlib.use('Agg')
        fig = self.visualizer.visualize()
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.axes), 0)
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_visualize_2d(self):
        import matplotlib
        matplotlib.use('Agg')
        fig = self.visualizer.visualize_2d_projection()
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_visualize_statistics(self):
        import matplotlib
        matplotlib.use('Agg')
        fig = self.visualizer.visualize_statistics()
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_color_scheme_density(self):
        config = OctreeConfig(voxel_size=0.5, max_depth=4, color_scheme='density')
        octree = OctreeMap(config)
        octree.build(self.points)
        viz = OctreeVisualizer(octree)
        import matplotlib
        matplotlib.use('Agg')
        fig = viz.visualize()
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_color_scheme_depth(self):
        config = OctreeConfig(voxel_size=0.5, max_depth=4, color_scheme='depth')
        octree = OctreeMap(config)
        octree.build(self.points)
        viz = OctreeVisualizer(octree)
        import matplotlib
        matplotlib.use('Agg')
        fig = viz.visualize()
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_show_borders(self):
        config = OctreeConfig(voxel_size=0.5, max_depth=4, show_borders=True)
        octree = OctreeMap(config)
        octree.build(self.points)
        viz = OctreeVisualizer(octree)
        import matplotlib
        matplotlib.use('Agg')
        fig = viz.visualize()
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_empty_octree_visualize(self):
        config = OctreeConfig(height_min=1000.0, height_max=2000.0)
        octree = OctreeMap(config)
        octree.build(self.points)
        viz = OctreeVisualizer(octree)
        import matplotlib
        matplotlib.use('Agg')
        fig = viz.visualize()
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_render_quality_low(self):
        config = OctreeConfig(voxel_size=0.5, max_depth=4, render_quality='low')
        octree = OctreeMap(config)
        octree.build(self.points)
        viz = OctreeVisualizer(octree)
        import matplotlib
        matplotlib.use('Agg')
        fig = viz.visualize()
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
