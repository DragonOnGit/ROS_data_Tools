# -*- coding: utf-8 -*-
"""
占据网格图生成模块
功能：基于点云数据生成2D占据网格图（全局/局部）

核心算法：
    将3D点云投影到2D平面（X-Y），按分辨率划分网格，
    统计每个网格单元内的点数，根据阈值判断占据/空闲/未知状态。

使用示例：
    from occupancy_grid import OccupancyGridMap, OccupancyGridConfig
    
    config = OccupancyGridConfig(resolution=0.05)
    ogm = OccupancyGridMap(config)
    ogm.build(points)
    ogm.visualize()
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import time


@dataclass
class OccupancyGridConfig:
    """占据网格图配置参数
    
    Attributes:
        resolution (float): 网格分辨率（米/格），值越小精度越高但内存占用越大
        height_min (float): 高度过滤下限（米），仅保留高于此值的点
        height_max (float): 高度过滤上限（米），仅保留低于此值的点
        occupancy_threshold (int): 占据阈值，网格内点数>=此值则判定为占据
        free_threshold (int): 空闲阈值，网格内点数<此值则判定为空闲
        unknown_value (int): 未知区域的值（默认-1）
        occupied_value (int): 占据区域的值（默认100）
        free_value (int): 空闲区域的值（默认0）
        local_range (float): 局部地图范围（米），以中心点为基准的正方形边长
        local_time_start (float): 局部地图时间过滤起始（秒）
        local_time_end (float): 局部地图时间过滤结束（秒）
    """
    resolution: float = 0.05
    height_min: float = float('-inf')
    height_max: float = float('inf')
    occupancy_threshold: int = 3
    free_threshold: int = 1
    unknown_value: int = -1
    occupied_value: int = 100
    free_value: int = 0
    local_range: float = 10.0
    local_time_start: float = 0.0
    local_time_end: float = float('inf')
    
    def __post_init__(self):
        self.resolution = np.clip(self.resolution, 0.01, 5.0)
        self.occupancy_threshold = max(1, self.occupancy_threshold)
        self.free_threshold = max(0, self.free_threshold)
        self.local_range = np.clip(self.local_range, 1.0, 1000.0)


class OccupancyGridMap:
    """占据网格图生成器
    
    基于点云数据生成2D占据网格图，支持：
    - 全局占据网格图（处理全部点云数据）
    - 局部占据网格图（以指定位置为中心，指定范围为区域）
    - 高度过滤（仅保留指定高度区间内的点）
    - 时间过滤（仅处理指定时间范围内的点）
    
    使用示例：
        >>> config = OccupancyGridConfig(resolution=0.05)
        >>> ogm = OccupancyGridMap(config)
        >>> ogm.build(points)
        >>> ogm.visualize()
    """
    
    def __init__(self, config: Optional[OccupancyGridConfig] = None):
        self.config = config if config is not None else OccupancyGridConfig()
        self.grid: Optional[np.ndarray] = None
        self.origin: np.ndarray = np.zeros(2)
        self.grid_size: Tuple[int, int] = (0, 0)
        self._build_time: float = 0.0
        self._filtered_count: int = 0
        self._total_count: int = 0
    
    @property
    def build_time(self) -> float:
        return self._build_time
    
    def build(self, points: np.ndarray, timestamps: Optional[np.ndarray] = None) -> None:
        """构建全局占据网格图
        
        Args:
            points: 点云数据，形状为(N, 3)或(N, 4)（含时间戳）
            timestamps: 可选的时间戳数组，形状为(N,)
        """
        start_time = time.time()
        
        if points is None or len(points) == 0:
            print("[WARN] 点云数据为空")
            return
        
        self._total_count = len(points)
        filtered = self._filter_points(points, timestamps)
        self._filtered_count = len(filtered)
        
        if len(filtered) == 0:
            print("[WARN] 过滤后无剩余点")
            self._build_time = time.time() - start_time
            return
        
        xy = filtered[:, :2]
        self._build_grid(xy)
        self._build_time = time.time() - start_time
        
        print(f"[OK] 全局占据网格图构建完成:")
        print(f"   原始点数: {self._total_count}")
        print(f"   过滤后点数: {self._filtered_count}")
        print(f"   网格尺寸: {self.grid_size[0]} x {self.grid_size[1]}")
        print(f"   分辨率: {self.config.resolution}m/格")
        print(f"   耗时: {self._build_time:.3f}s")
    
    def build_local(self, points: np.ndarray, center: np.ndarray,
                    timestamps: Optional[np.ndarray] = None) -> None:
        """构建局部占据网格图
        
        Args:
            points: 点云数据，形状为(N, 3)
            center: 中心点坐标 (x, y)
            timestamps: 可选的时间戳数组
        """
        start_time = time.time()
        
        if points is None or len(points) == 0:
            print("[WARN] 点云数据为空")
            return
        
        self._total_count = len(points)
        filtered = self._filter_points(points, timestamps)
        self._filtered_count = len(filtered)
        
        if len(filtered) == 0:
            print("[WARN] 过滤后无剩余点")
            self._build_time = time.time() - start_time
            return
        
        xy = filtered[:, :2]
        half = self.config.local_range / 2.0
        
        mask = (
            (xy[:, 0] >= center[0] - half) & (xy[:, 0] <= center[0] + half) &
            (xy[:, 1] >= center[1] - half) & (xy[:, 1] <= center[1] + half)
        )
        local_xy = xy[mask]
        
        if len(local_xy) == 0:
            print("[WARN] 局部范围内无点云数据")
            self.grid = np.full(
                (int(self.config.local_range / self.config.resolution),) * 2,
                self.config.unknown_value, dtype=np.int8
            )
            self.origin = np.array([center[0] - half, center[1] - half])
            self.grid_size = self.grid.shape
            self._build_time = time.time() - start_time
            return
        
        self._build_grid(local_xy)
        self._build_time = time.time() - start_time
        
        print(f"[OK] 局部占据网格图构建完成:")
        print(f"   中心: ({center[0]:.2f}, {center[1]:.2f})")
        print(f"   范围: {self.config.local_range}m x {self.config.local_range}m")
        print(f"   局部点数: {len(local_xy)}")
        print(f"   网格尺寸: {self.grid_size[0]} x {self.grid_size[1]}")
        print(f"   耗时: {self._build_time:.3f}s")
    
    def _filter_points(self, points: np.ndarray,
                       timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """过滤点云数据（高度+时间）"""
        if len(points) == 0:
            return points
        
        mask = np.ones(len(points), dtype=bool)
        
        z = points[:, 2]
        mask &= (z >= self.config.height_min) & (z <= self.config.height_max)
        
        if timestamps is not None and len(timestamps) == len(points):
            mask &= (timestamps >= self.config.local_time_start)
            mask &= (timestamps <= self.config.local_time_end)
        
        return points[mask]
    
    def _build_grid(self, xy: np.ndarray) -> None:
        """从2D点云构建占据网格"""
        if len(xy) == 0:
            return
        
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        
        margin = self.config.resolution
        x_min -= margin
        y_min -= margin
        x_max += margin
        y_max += margin
        
        self.origin = np.array([x_min, y_min])
        
        width = int(np.ceil((x_max - x_min) / self.config.resolution))
        height = int(np.ceil((y_max - y_min) / self.config.resolution))
        
        self.grid = np.full((height, width), self.config.unknown_value, dtype=np.int8)
        
        gx = ((xy[:, 0] - x_min) / self.config.resolution).astype(int)
        gy = ((xy[:, 1] - y_min) / self.config.resolution).astype(int)
        
        valid = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
        gx = gx[valid]
        gy = gy[valid]
        
        counts = np.zeros((height, width), dtype=np.int32)
        np.add.at(counts, (gy, gx), 1)
        
        self.grid[counts >= self.config.occupancy_threshold] = self.config.occupied_value
        self.grid[(counts > 0) & (counts < self.config.occupancy_threshold)] = self.config.free_value
        
        self.grid_size = (height, width)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if self.grid is None:
            return {}
        
        total = self.grid.size
        occupied = np.sum(self.grid == self.config.occupied_value)
        free = np.sum(self.grid == self.config.free_value)
        unknown = np.sum(self.grid == self.config.unknown_value)
        
        return {
            'total_cells': int(total),
            'occupied_cells': int(occupied),
            'free_cells': int(free),
            'unknown_cells': int(unknown),
            'occupancy_rate': float(occupied / total * 100) if total > 0 else 0,
            'grid_width': self.grid_size[1],
            'grid_height': self.grid_size[0],
            'resolution': self.config.resolution,
            'build_time': self._build_time,
            'total_points': self._total_count,
            'filtered_points': self._filtered_count,
        }
    
    def visualize(self, fig=None, title: str = '占据网格图') -> 'Figure':
        """可视化占据网格图
        
        Args:
            fig: 可选的matplotlib Figure
            title: 图表标题
            
        Returns:
            Figure: matplotlib Figure对象
        """
        from matplotlib.figure import Figure
        from matplotlib.colors import ListedColormap
        
        if self.grid is None:
            print("[WARN] 无网格数据可可视化")
            fig = fig or Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_title('无数据')
            return fig
        
        fig = fig or Figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111)
        
        colors = ['#CCCCCC', '#FFFFFF', '#000000']
        cmap = ListedColormap(colors)
        
        display_grid = self.grid.copy().astype(float)
        display_grid[display_grid == self.config.unknown_value] = 0
        display_grid[display_grid == self.config.free_value] = 1
        display_grid[display_grid == self.config.occupied_value] = 2
        
        extent = [
            self.origin[0],
            self.origin[0] + self.grid_size[1] * self.config.resolution,
            self.origin[1],
            self.origin[1] + self.grid_size[0] * self.config.resolution,
        ]
        
        ax.imshow(display_grid, cmap=cmap, origin='lower', extent=extent,
                 interpolation='nearest', vmin=0, vmax=2)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(False)
        
        stats = self.get_statistics()
        info_text = (
            f"网格: {stats.get('grid_width', 0)}x{stats.get('grid_height', 0)} | "
            f"分辨率: {stats.get('resolution', 0):.3f}m | "
            f"占据率: {stats.get('occupancy_rate', 0):.1f}%"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        return fig
    
    def export_grid(self, filepath: str) -> bool:
        """导出网格数据"""
        if self.grid is None:
            print("[WARN] 无网格数据可导出")
            return False
        try:
            np.savez(filepath, grid=self.grid, origin=self.origin,
                    resolution=self.config.resolution,
                    grid_size=np.array(self.grid_size))
            print(f"[OK] 网格数据已导出: {filepath}")
            return True
        except Exception as e:
            print(f"[ERROR] 导出失败: {str(e)}")
            return False
