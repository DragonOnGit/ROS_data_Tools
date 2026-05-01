# -*- coding: utf-8 -*-
"""
八叉树地图可视化模块
功能：基于matplotlib实现八叉树体素的3D可视化，支持多种颜色映射和渲染配置

使用示例：
    from octree_map import OctreeMap, OctreeConfig
    from octree_visualizer import OctreeVisualizer
    
    octree = OctreeMap(config)
    octree.build(points)
    
    visualizer = OctreeVisualizer(octree)
    fig = visualizer.visualize()
"""

import numpy as np
from typing import Optional, List, Dict
from matplotlib.figure import Figure
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from octree_map import OctreeMap, OctreeConfig


class OctreeVisualizer:
    """八叉树地图可视化器
    
    提供多种可视化方式：
    - 体素3D渲染（支持颜色映射、透明度、边界线）
    - 2D投影俯视图
    - 统计图表
    
    Attributes:
        octree (OctreeMap): 八叉树地图实例
        config (OctreeConfig): 配置参数
    """
    
    _QUALITY_SETTINGS = {
        'low': {'max_voxels': 2000, 'dpi': 80},
        'medium': {'max_voxels': 8000, 'dpi': 100},
        'high': {'max_voxels': 20000, 'dpi': 150},
    }
    
    _COLORMAPS = {
        'height': cm.viridis,
        'density': cm.hot,
        'depth': cm.cool,
        'custom': cm.jet,
    }
    
    def __init__(self, octree: OctreeMap):
        """初始化可视化器
        
        Args:
            octree: 已构建的八叉树地图实例
        """
        self.octree = octree
        self.config = octree.config
    
    def visualize(self, fig: Optional[Figure] = None) -> Figure:
        """生成3D体素可视化图
        
        Args:
            fig: 可选的Figure对象，为None时创建新图
            
        Returns:
            Figure: matplotlib Figure对象
        """
        voxels = self.octree.get_leaf_voxels()
        if not voxels:
            print("[WARN] 无体素数据可可视化")
            fig = fig or Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('无数据')
            return fig
        
        quality = self._QUALITY_SETTINGS.get(
            self.config.render_quality, self._QUALITY_SETTINGS['medium']
        )
        max_voxels = quality['max_voxels']
        
        # 按密度排序，优先显示高密度体素
        sorted_voxels = sorted(voxels, key=lambda v: v['point_count'], reverse=True)
        if len(sorted_voxels) > max_voxels:
            print(f"[WARN] 体素数量({len(sorted_voxels)})超过渲染上限({max_voxels})，仅显示密度最高的{max_voxels}个体素")
            sorted_voxels = sorted_voxels[:max_voxels]
        
        fig = fig or Figure(figsize=(12, 9), dpi=quality['dpi'])
        ax = fig.add_subplot(111, projection='3d')
        
        # 计算颜色映射
        colors = self._compute_colors(sorted_voxels)
        
        # 渲染体素
        alpha = self.config.transparency
        show_borders = self.config.show_borders
        
        for i, voxel in enumerate(sorted_voxels):
            self._draw_voxel(ax, voxel['center'], voxel['size'],
                           colors[i], alpha, show_borders)
        
        # 设置坐标轴
        centers = np.array([v['center'] for v in sorted_voxels])
        if len(centers) > 0:
            margin = 0.5
            ax.set_xlim(centers[:, 0].min() - margin, centers[:, 0].max() + margin)
            ax.set_ylim(centers[:, 1].min() - margin, centers[:, 1].max() + margin)
            ax.set_zlim(centers[:, 2].min() - margin, centers[:, 2].max() + margin)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        
        scheme_name = {'height': '高度', 'density': '密度', 'depth': '深度', 'custom': '自定义'}
        ax.set_title(f'八叉树地图 - {scheme_name.get(self.config.color_scheme, "")}映射',
                    fontsize=13, fontweight='bold')
        
        # 添加颜色条
        self._add_colorbar(fig, sorted_voxels)
        
        ax.view_init(elev=30, azim=45)
        fig.tight_layout()
        
        return fig
    
    def visualize_2d_projection(self, fig: Optional[Figure] = None) -> Figure:
        """生成2D俯视投影图
        
        Args:
            fig: 可选的Figure对象
            
        Returns:
            Figure: matplotlib Figure对象
        """
        voxels = self.octree.get_leaf_voxels()
        if not voxels:
            print("[WARN] 无体素数据可可视化")
            fig = fig or Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_title('无数据')
            return fig
        
        fig = fig or Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        centers = np.array([v['center'] for v in voxels])
        sizes = np.array([v['size'] for v in voxels])
        counts = np.array([v['point_count'] for v in voxels])
        
        colors = self._compute_colors(voxels)
        
        for i, voxel in enumerate(voxels):
            half = voxel['size'] / 2.0
            rect = plt.Rectangle(
                (voxel['center'][0] - half, voxel['center'][1] - half),
                voxel['size'], voxel['size'],
                facecolor=colors[i], edgecolor='gray',
                alpha=self.config.transparency, linewidth=0.3
            )
            ax.add_patch(rect)
        
        margin = 0.5
        ax.set_xlim(centers[:, 0].min() - margin, centers[:, 0].max() + margin)
        ax.set_ylim(centers[:, 1].min() - margin, centers[:, 1].max() + margin)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title('八叉树地图 - 2D俯视投影', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        return fig
    
    def visualize_statistics(self, fig: Optional[Figure] = None) -> Figure:
        """生成统计图表
        
        Args:
            fig: 可选的Figure对象
            
        Returns:
            Figure: matplotlib Figure对象
        """
        voxels = self.octree.get_leaf_voxels()
        stats = self.octree.get_statistics()
        
        fig = fig or Figure(figsize=(14, 5), dpi=100)
        
        # 子图1: 深度分布
        ax1 = fig.add_subplot(131)
        depths = [v['depth'] for v in voxels]
        if depths:
            ax1.hist(depths, bins=max(depths) - min(depths) + 1, color='steelblue', edgecolor='white')
        ax1.set_title('深度分布', fontsize=11)
        ax1.set_xlabel('深度')
        ax1.set_ylabel('体素数量')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 点数分布
        ax2 = fig.add_subplot(132)
        counts = [v['point_count'] for v in voxels]
        if counts:
            ax2.hist(counts, bins=30, color='coral', edgecolor='white')
        ax2.set_title('每体素点数分布', fontsize=11)
        ax2.set_xlabel('点数')
        ax2.set_ylabel('体素数量')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 高度分布
        ax3 = fig.add_subplot(133)
        z_values = [v['z'] for v in voxels]
        if z_values:
            ax3.hist(z_values, bins=30, color='seagreen', edgecolor='white')
        ax3.set_title('高度分布', fontsize=11)
        ax3.set_xlabel('高度 (m)')
        ax3.set_ylabel('体素数量')
        ax3.grid(True, alpha=0.3)
        
        fig.suptitle('八叉树统计分析', fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        return fig
    
    def _compute_colors(self, voxels: List[Dict]) -> List[np.ndarray]:
        """根据颜色映射方案计算体素颜色
        
        Args:
            voxels: 体素列表
            
        Returns:
            List[np.ndarray]: RGBA颜色数组
        """
        scheme = self.config.color_scheme
        cmap = self._COLORMAPS.get(scheme, cm.viridis)
        
        if scheme == 'height':
            values = np.array([v['z'] for v in voxels])
        elif scheme == 'density':
            values = np.array([v['density'] for v in voxels])
        elif scheme == 'depth':
            values = np.array([v['depth'] for v in voxels])
        else:
            values = np.array([v['z'] for v in voxels])
        
        if len(values) == 0:
            return []
        
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-10:
            norm_values = np.zeros_like(values)
        else:
            norm_values = (values - vmin) / (vmax - vmin)
        
        colors = [cmap(v) for v in norm_values]
        return colors
    
    def _draw_voxel(self, ax, center: np.ndarray, size: float,
                    color: np.ndarray, alpha: float, show_borders: bool) -> None:
        """绘制单个体素（立方体）
        
        Args:
            ax: 3D坐标轴
            center: 体素中心坐标
            size: 体素边长
            color: RGBA颜色
            alpha: 透明度
            show_borders: 是否显示边界线
        """
        r = size / 2.0
        x, y, z = center
        
        # 6个面的顶点
        vertices = [
            [[x-r, y-r, z-r], [x+r, y-r, z-r], [x+r, y+r, z-r], [x-r, y+r, z-r]],
            [[x-r, y-r, z+r], [x+r, y-r, z+r], [x+r, y+r, z+r], [x-r, y+r, z+r]],
            [[x-r, y-r, z-r], [x+r, y-r, z-r], [x+r, y-r, z+r], [x-r, y-r, z+r]],
            [[x-r, y+r, z-r], [x+r, y+r, z-r], [x+r, y+r, z+r], [x-r, y+r, z+r]],
            [[x-r, y-r, z-r], [x-r, y+r, z-r], [x-r, y+r, z+r], [x-r, y-r, z+r]],
            [[x+r, y-r, z-r], [x+r, y+r, z-r], [x+r, y+r, z+r], [x+r, y-r, z+r]],
        ]
        
        facecolor = (*color[:3], alpha)
        edgecolor = (0.3, 0.3, 0.3, 0.5) if show_borders else (0, 0, 0, 0)
        
        poly = Poly3DCollection(vertices, alpha=alpha)
        poly.set_facecolor([facecolor])
        poly.set_edgecolor([edgecolor])
        ax.add_collection3d(poly)
    
    def _add_colorbar(self, fig: Figure, voxels: List[Dict]) -> None:
        """添加颜色条
        
        Args:
            fig: Figure对象
            voxels: 体素列表
        """
        scheme = self.config.color_scheme
        cmap = self._COLORMAPS.get(scheme, cm.viridis)
        
        if scheme == 'height':
            values = [v['z'] for v in voxels]
            label = '高度 (m)'
        elif scheme == 'density':
            values = [v['density'] for v in voxels]
            label = '密度 (点/m³)'
        elif scheme == 'depth':
            values = [v['depth'] for v in voxels]
            label = '深度'
        else:
            values = [v['z'] for v in voxels]
            label = '高度 (m)'
        
        if not values:
            return
        
        norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=fig.axes[0], shrink=0.6, pad=0.1)
        cbar.set_label(label, fontsize=9)
