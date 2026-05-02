# -*- coding: utf-8 -*-
"""
占据网格图生成模块
功能：基于点云数据生成2D占据网格图（全局/局部）

核心算法：
    采用射线投射（Ray-Casting）算法：
    从传感器位置向每个点云数据点投射射线，
    射线经过的网格标记为"空闲"（传感器看穿的区域），
    射线终点标记为"占据"（传感器击中的障碍物），
    未被任何射线经过的网格保持"未知"状态。

使用示例：
    from occupancy_grid import OccupancyGridMap, OccupancyGridConfig
    
    config = OccupancyGridConfig(resolution=0.05)
    ogm = OccupancyGridMap(config)
    ogm.build(points, sensor_positions=sensor_pos)
    ogm.visualize()
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import time


class CoordinateTransformer:
    """坐标系转换器
    
    处理无人机坐标系与地图坐标系之间的转换。
    当两个坐标系的x轴和y轴方向完全相反时，
    需要将无人机位置取反后映射到地图坐标系。
    
    转换公式：
        map_x = -drone_x
        map_y = -drone_y
    
    使用示例：
        transformer = CoordinateTransformer(flip_x=True, flip_y=True)
        map_pos = transformer.transform_point(drone_pos)
        map_traj = transformer.transform_trajectory(drone_xy)
    """
    
    def __init__(self, flip_x: bool = True, flip_y: bool = True,
                 offset_x: float = 0.0, offset_y: float = 0.0):
        """
        Args:
            flip_x: 是否翻转x轴
            flip_y: 是否翻转y轴
            offset_x: x轴偏移量（翻转后叠加）
            offset_y: y轴偏移量（翻转后叠加）
        """
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.offset_x = offset_x
        self.offset_y = offset_y
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """转换单个点坐标
        
        Args:
            point: 坐标 (x, y) 或 (x, y, z)
            
        Returns:
            转换后的坐标
        """
        result = point.copy()
        if self.flip_x:
            result[0] = -result[0] + self.offset_x
        else:
            result[0] = result[0] + self.offset_x
        if self.flip_y:
            result[1] = -result[1] + self.offset_y
        else:
            result[1] = result[1] + self.offset_y
        return result
    
    def transform_trajectory(self, xy: np.ndarray) -> np.ndarray:
        """转换轨迹坐标数组
        
        Args:
            xy: 坐标数组，形状为(N, 2)或(N, 3)
            
        Returns:
            转换后的坐标数组
        """
        result = xy.copy()
        if self.flip_x:
            result[:, 0] = -result[:, 0] + self.offset_x
        else:
            result[:, 0] = result[:, 0] + self.offset_x
        if self.flip_y:
            result[:, 1] = -result[:, 1] + self.offset_y
        else:
            result[:, 1] = result[:, 1] + self.offset_y
        return result


@dataclass
class TrajectoryOverlay:
    """轨迹叠加配置
    
    Attributes:
        actual_xy: 实际飞行轨迹坐标 (N, 2)
        expected_xy: 期望轨迹坐标 (M, 2)
        show_actual: 是否显示实际轨迹
        show_expected: 是否显示期望轨迹
        show_all: 全局轨迹显示总开关
        actual_color: 实际轨迹颜色
        expected_color: 期望轨迹颜色
        actual_linewidth: 实际轨迹线宽
        expected_linewidth: 期望轨迹线宽
        arrow_interval: 方向箭头间隔（米）
        arrow_length: 方向箭头长度（像素），3倍线宽=6像素
    """
    actual_xy: Optional[np.ndarray] = None
    expected_xy: Optional[np.ndarray] = None
    show_actual: bool = True
    show_expected: bool = True
    show_all: bool = True
    actual_color: tuple = (0, 0, 1.0)
    expected_color: tuple = (1.0, 0, 0)
    actual_linewidth: float = 2.0
    expected_linewidth: float = 2.0
    arrow_interval: float = 5.0
    arrow_length: float = 6.0


@dataclass
class OccupancyGridConfig:
    """占据网格图配置参数
    
    Attributes:
        resolution (float): 网格分辨率（米/格），值越小精度越高但内存占用越大
        height_min (float): 高度过滤下限（米），仅保留高于此值的点
        height_max (float): 高度过滤上限（米），仅保留低于此值的点
        occupancy_threshold (int): 占据阈值，网格内点数>=此值则判定为占据
        unknown_value (int): 未知区域的值（默认-1）
        occupied_value (int): 占据区域的值（默认100）
        free_value (int): 空闲区域的值（默认0）
        local_range (float): 局部地图范围（米），以中心点为基准的正方形边长
        local_time_start (float): 局部地图时间过滤起始（秒）
        local_time_end (float): 局部地图时间过滤结束（秒）
        use_raycast (bool): 是否使用射线投射算法标记空闲区域
        ray_max_dist (float): 射线最大距离（米），超过此距离的点不参与射线投射
    """
    resolution: float = 0.05
    height_min: float = float('-inf')
    height_max: float = float('inf')
    occupancy_threshold: int = 3
    unknown_value: int = -1
    occupied_value: int = 100
    free_value: int = 0
    local_range: float = 10.0
    local_time_start: float = 0.0
    local_time_end: float = float('inf')
    use_raycast: bool = True
    ray_max_dist: float = 50.0
    
    def __post_init__(self):
        self.resolution = np.clip(self.resolution, 0.01, 5.0)
        self.occupancy_threshold = max(1, self.occupancy_threshold)
        self.local_range = np.clip(self.local_range, 1.0, 1000.0)
        self.ray_max_dist = np.clip(self.ray_max_dist, 1.0, 500.0)


class OccupancyGridMap:
    """占据网格图生成器
    
    基于点云数据生成2D占据网格图，支持：
    - 全局占据网格图（处理全部点云数据）
    - 局部占据网格图（以指定位置为中心，指定范围为区域）
    - 射线投射算法标记空闲区域
    - 高度过滤（仅保留指定高度区间内的点）
    - 时间过滤（仅处理指定时间范围内的点）
    
    使用示例：
        >>> config = OccupancyGridConfig(resolution=0.05)
        >>> ogm = OccupancyGridMap(config)
        >>> ogm.build(points, sensor_positions=sensor_pos)
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
        self._local_center: Optional[np.ndarray] = None
        self._local_range: float = 0.0
    
    @property
    def build_time(self) -> float:
        return self._build_time
    
    @property
    def local_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取局部地图的边界坐标（用于在全局图上标识）"""
        if self._local_center is None:
            return None
        half = self._local_range / 2.0
        min_pt = np.array([self._local_center[0] - half, self._local_center[1] - half])
        max_pt = np.array([self._local_center[0] + half, self._local_center[1] + half])
        return (min_pt, max_pt)
    
    def build(self, points: np.ndarray, timestamps: Optional[np.ndarray] = None,
              sensor_positions: Optional[np.ndarray] = None) -> None:
        """构建全局占据网格图
        
        Args:
            points: 点云数据，形状为(N, 3)或(N, 4)（含时间戳）
            timestamps: 可选的时间戳数组，形状为(N,)
            sensor_positions: 传感器位置数组，形状为(M, 2)或(M, 3)
                              如果提供，将使用射线投射标记空闲区域
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
        self._build_grid(xy, sensor_positions)
        self._build_time = time.time() - start_time
        
        stats = self.get_statistics()
        print(f"[OK] 全局占据网格图构建完成:")
        print(f"   原始点数: {self._total_count}")
        print(f"   过滤后点数: {self._filtered_count}")
        print(f"   网格尺寸: {self.grid_size[0]} x {self.grid_size[1]}")
        print(f"   分辨率: {self.config.resolution}m/格")
        print(f"   占据率: {stats['occupancy_rate']:.1f}%")
        print(f"   空闲率: {stats['free_rate']:.1f}%")
        print(f"   耗时: {self._build_time:.3f}s")
    
    def build_local(self, points: np.ndarray, center: np.ndarray,
                    timestamps: Optional[np.ndarray] = None,
                    sensor_position: Optional[np.ndarray] = None) -> None:
        """构建局部占据网格图（从原始点云数据构建）
        
        Args:
            points: 点云数据，形状为(N, 3)
            center: 中心点坐标 (x, y)
            timestamps: 可选的时间戳数组
            sensor_position: 传感器位置 (x, y)，默认使用center
        """
        start_time = time.time()
        
        if points is None or len(points) == 0:
            print("[WARN] 点云数据为空")
            return
        
        self._local_center = center.copy()
        self._local_range = self.config.local_range
        
        self._total_count = len(points)
        filtered = self._filter_points(points, timestamps)
        self._filtered_count = len(filtered)
        
        half = self.config.local_range / 2.0
        
        if len(filtered) == 0:
            print("[WARN] 过滤后无剩余点")
            w = int(np.ceil(self.config.local_range / self.config.resolution))
            h = int(np.ceil(self.config.local_range / self.config.resolution))
            self.grid = np.full((h, w), self.config.unknown_value, dtype=np.int8)
            self.origin = np.array([center[0] - half, center[1] - half])
            self.grid_size = (h, w)
            self._build_time = time.time() - start_time
            return
        
        xy = filtered[:, :2]
        mask = (
            (xy[:, 0] >= center[0] - half) & (xy[:, 0] <= center[0] + half) &
            (xy[:, 1] >= center[1] - half) & (xy[:, 1] <= center[1] + half)
        )
        local_xy = xy[mask]
        
        if len(local_xy) == 0:
            print("[WARN] 局部范围内无点云数据")
            w = int(np.ceil(self.config.local_range / self.config.resolution))
            h = int(np.ceil(self.config.local_range / self.config.resolution))
            self.grid = np.full((h, w), self.config.unknown_value, dtype=np.int8)
            self.origin = np.array([center[0] - half, center[1] - half])
            self.grid_size = (h, w)
            self._build_time = time.time() - start_time
            return
        
        if sensor_position is None:
            sensor_position = center
        
        self._build_grid(local_xy, sensor_positions=sensor_position.reshape(1, -1))
        self._build_time = time.time() - start_time
        
        stats = self.get_statistics()
        print(f"[OK] 局部占据网格图构建完成:")
        print(f"   中心: ({center[0]:.2f}, {center[1]:.2f})")
        print(f"   范围: {self.config.local_range}m x {self.config.local_range}m")
        print(f"   局部点数: {len(local_xy)}")
        print(f"   网格尺寸: {self.grid_size[0]} x {self.grid_size[1]}")
        print(f"   占据率: {stats['occupancy_rate']:.1f}%")
        print(f"   空闲率: {stats['free_rate']:.1f}%")
        print(f"   耗时: {self._build_time:.3f}s")
    
    def extract_local_from_global(self, global_ogm: 'OccupancyGridMap',
                                  center: np.ndarray,
                                  local_range: Optional[float] = None) -> bool:
        """从全局占据网格图中提取局部区域
        
        实现"时间→位置→区域地图"三级映射的最后两步：
        基于给定的中心坐标，从全局地图中提取指定范围的占据图数据，
        将提取的区域地图内容准确渲染至局部坐标系。
        
        Args:
            global_ogm: 全局占据网格图对象
            center: 局部地图中心坐标 (x, y)，通常为无人机在地图坐标系中的位置
            local_range: 局部范围（米），None则使用config中的值
            
        Returns:
            bool: 提取是否成功
        """
        start_time = time.time()
        
        if global_ogm is None or global_ogm.grid is None:
            print("[WARN] 全局占据网格图不存在")
            return False
        
        if local_range is None:
            local_range = self.config.local_range
        
        self._local_center = center.copy()
        self._local_range = local_range
        
        half = local_range / 2.0
        local_x_min = center[0] - half
        local_x_max = center[0] + half
        local_y_min = center[1] - half
        local_y_max = center[1] + half
        
        g_res = global_ogm.config.resolution
        g_origin = global_ogm.origin
        g_grid = global_ogm.grid
        g_height, g_width = g_grid.shape
        
        g_col_min = int(np.floor((local_x_min - g_origin[0]) / g_res))
        g_col_max = int(np.ceil((local_x_max - g_origin[0]) / g_res))
        g_row_min = int(np.floor((local_y_min - g_origin[1]) / g_res))
        g_row_max = int(np.ceil((local_y_max - g_origin[1]) / g_res))
        
        g_col_start = max(0, g_col_min)
        g_col_end = min(g_width, g_col_max)
        g_row_start = max(0, g_row_min)
        g_row_end = min(g_height, g_row_max)
        
        l_width = int(np.ceil(local_range / self.config.resolution))
        l_height = int(np.ceil(local_range / self.config.resolution))
        
        self.grid = np.full((l_height, l_width), self.config.unknown_value, dtype=np.int8)
        self.origin = np.array([local_x_min, local_y_min])
        
        if g_col_start >= g_col_end or g_row_start >= g_row_end:
            self.grid_size = (l_height, l_width)
            self._build_time = time.time() - start_time
            print("[WARN] 局部区域超出全局地图范围")
            return True
        
        src = g_grid[g_row_start:g_row_end, g_col_start:g_col_end]
        
        l_col_start = g_col_start - g_col_min
        l_row_start = g_row_start - g_row_min
        l_col_end = l_col_start + src.shape[1]
        l_row_end = l_row_start + src.shape[0]
        
        l_col_start = max(0, l_col_start)
        l_row_start = max(0, l_row_start)
        l_col_end = min(l_width, l_col_end)
        l_row_end = min(l_height, l_row_end)
        
        s_col = l_col_start - (g_col_start - g_col_min)
        s_row = l_row_start - (g_row_start - g_row_min)
        
        copy_h = l_row_end - l_row_start
        copy_w = l_col_end - l_col_start
        
        if copy_h > 0 and copy_w > 0 and s_row >= 0 and s_col >= 0:
            src_slice = src[s_row:s_row+copy_h, s_col:s_col+copy_w]
            self.grid[l_row_start:l_row_end, l_col_start:l_col_end] = src_slice
        
        self.grid_size = (l_height, l_width)
        self._build_time = time.time() - start_time
        
        stats = self.get_statistics()
        print(f"[OK] 局部占据网格图提取完成:")
        print(f"   中心: ({center[0]:.2f}, {center[1]:.2f})")
        print(f"   范围: {local_range}m x {local_range}m")
        print(f"   网格尺寸: {self.grid_size[0]} x {self.grid_size[1]}")
        print(f"   占据率: {stats['occupancy_rate']:.1f}%")
        print(f"   空闲率: {stats['free_rate']:.1f}%")
        print(f"   耗时: {self._build_time:.3f}s")
        return True
    
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
    
    def _build_grid(self, xy: np.ndarray,
                    sensor_positions: Optional[np.ndarray] = None) -> None:
        """从2D点云构建占据网格（含射线投射）
        
        Args:
            xy: 2D点云坐标，形状为(N, 2)
            sensor_positions: 传感器位置，形状为(M, 2)
        """
        if len(xy) == 0:
            return
        
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        
        if sensor_positions is not None and len(sensor_positions) > 0:
            sp = sensor_positions[:, :2]
            x_min = min(x_min, sp[:, 0].min())
            y_min = min(y_min, sp[:, 1].min())
            x_max = max(x_max, sp[:, 0].max())
            y_max = max(y_max, sp[:, 1].max())
        
        margin = self.config.resolution * 2
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
        
        occupied_mask = counts >= self.config.occupancy_threshold
        self.grid[occupied_mask] = self.config.occupied_value
        
        if self.config.use_raycast and sensor_positions is not None and len(sensor_positions) > 0:
            self._apply_raycast(xy, sensor_positions, occupied_mask, height, width, x_min, y_min)
        else:
            self.grid[(counts > 0) & (~occupied_mask)] = self.config.free_value
        
        self.grid_size = (height, width)
    
    def _apply_raycast(self, xy: np.ndarray, sensor_positions: np.ndarray,
                       occupied_mask: np.ndarray, height: int, width: int,
                       x_min: float, y_min: float) -> None:
        """应用射线投射算法标记空闲区域
        
        对每个传感器位置，向每个点云数据点投射射线，
        射线经过的网格标记为空闲，射线终点标记为占据。
        
        Args:
            xy: 2D点云坐标
            sensor_positions: 传感器位置数组
            occupied_mask: 已标记为占据的网格掩码
            height, width: 网格尺寸
            x_min, y_min: 网格原点坐标
        """
        res = self.config.resolution
        max_dist_sq = self.config.ray_max_dist ** 2
        free_grid = np.zeros((height, width), dtype=np.int32)
        
        sp = sensor_positions[:, :2]
        
        n_points = len(xy)
        step = max(1, n_points // 5000)
        sampled_xy = xy[::step]
        
        for sensor in sp:
            sx = int((sensor[0] - x_min) / res)
            sy = int((sensor[1] - y_min) / res)
            
            if sx < 0 or sx >= width or sy < 0 or sy >= height:
                continue
            
            for pt in sampled_xy:
                dx = pt[0] - sensor[0]
                dy = pt[1] - sensor[1]
                dist_sq = dx * dx + dy * dy
                
                if dist_sq > max_dist_sq or dist_sq < (res * res):
                    continue
                
                ex = int((pt[0] - x_min) / res)
                ey = int((pt[1] - y_min) / res)
                
                ex = np.clip(ex, 0, width - 1)
                ey = np.clip(ey, 0, height - 1)
                
                cells = self._bresenham_line(sx, sy, ex, ey)
                
                for ci, (cx, cy) in enumerate(cells):
                    if 0 <= cx < width and 0 <= cy < height:
                        if ci < len(cells) - 1:
                            free_grid[cy, cx] += 1
                        else:
                            pass
            
            free_grid[sy, sx] += 1
        
        free_threshold = max(1, len(sp) // 2)
        free_mask = (free_grid >= free_threshold) & (~occupied_mask)
        self.grid[free_mask] = self.config.free_value
        
        print(f"   射线投射完成: 传感器数={len(sp)}, 采样点数={len(sampled_xy)}")
    
    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham直线算法，返回线段经过的所有网格坐标
        
        Args:
            x0, y0: 起点网格坐标
            x1, y1: 终点网格坐标
            
        Returns:
            网格坐标列表 [(x, y), ...]
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        max_steps = dx + dy + 2
        step = 0
        
        while step < max_steps:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            step += 1
        
        return cells
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if self.grid is None:
            return {}
        
        total = self.grid.size
        occupied = int(np.sum(self.grid == self.config.occupied_value))
        free = int(np.sum(self.grid == self.config.free_value))
        unknown = int(np.sum(self.grid == self.config.unknown_value))
        
        return {
            'total_cells': int(total),
            'occupied_cells': occupied,
            'free_cells': free,
            'unknown_cells': unknown,
            'occupancy_rate': float(occupied / total * 100) if total > 0 else 0,
            'free_rate': float(free / total * 100) if total > 0 else 0,
            'grid_width': self.grid_size[1],
            'grid_height': self.grid_size[0],
            'resolution': self.config.resolution,
            'build_time': self._build_time,
            'total_points': self._total_count,
            'filtered_points': self._filtered_count,
        }
    
    def visualize(self, fig=None, title: str = '占据网格图',
                  local_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                  center: Optional[np.ndarray] = None,
                  trajectory_overlay: Optional['TrajectoryOverlay'] = None,
                  is_local: bool = False) -> 'Figure':
        """可视化占据网格图
        
        Args:
            fig: 可选的matplotlib Figure
            title: 图表标题
            local_bounds: 局部区域边界 (min_xy, max_xy)，用于在全局图上标识
            center: 中心点坐标，用于标识无人机位置
            trajectory_overlay: 轨迹叠加配置
            is_local: 是否为局部图（影响箭头绘制）
            
        Returns:
            Figure: matplotlib Figure对象
        """
        from matplotlib.figure import Figure
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Rectangle, FancyArrowPatch
        
        if self.grid is None:
            print("[WARN] 无网格数据可可视化")
            fig = fig or Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_title('无数据')
            return fig
        
        fig = fig or Figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111)
        
        colors = ['#D0D0D0', '#FFFFFF', '#1A1A1A']
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
        
        if local_bounds is not None:
            min_xy, max_xy = local_bounds
            rect_w = max_xy[0] - min_xy[0]
            rect_h = max_xy[1] - min_xy[1]
            rect = Rectangle(
                (min_xy[0], min_xy[1]), rect_w, rect_h,
                linewidth=2.5, edgecolor='#FF4444', facecolor='#FF4444',
                alpha=0.15, linestyle='--', zorder=10
            )
            ax.add_patch(rect)
            rect_border = Rectangle(
                (min_xy[0], min_xy[1]), rect_w, rect_h,
                linewidth=2.5, edgecolor='#FF4444', facecolor='none',
                linestyle='--', zorder=11
            )
            ax.add_patch(rect_border)
            cx = (min_xy[0] + max_xy[0]) / 2
            cy = (min_xy[1] + max_xy[1]) / 2
            ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2.5, zorder=12)
            ax.annotate('局部区域', xy=(min_xy[0], max_xy[1]),
                       fontsize=9, color='#FF4444', fontweight='bold',
                       xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#FF4444'))
        
        if center is not None:
            ax.plot(center[0], center[1], 'r^', markersize=12,
                   markeredgecolor='darkred', markeredgewidth=1.5, zorder=15,
                   label='无人机位置')
        
        if trajectory_overlay is not None and trajectory_overlay.show_all:
            self._draw_trajectories(ax, trajectory_overlay, extent, is_local)
        
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        if center is not None or (trajectory_overlay is not None and trajectory_overlay.show_all):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right', fontsize=9)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(False)
        
        stats = self.get_statistics()
        info_text = (
            f"网格: {stats.get('grid_width', 0)}x{stats.get('grid_height', 0)} | "
            f"分辨率: {stats.get('resolution', 0):.3f}m | "
            f"占据: {stats.get('occupancy_rate', 0):.1f}% | "
            f"空闲: {stats.get('free_rate', 0):.1f}%"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        fig.tight_layout()
        return fig
    
    def _draw_trajectories(self, ax, overlay: 'TrajectoryOverlay',
                           extent: list, is_local: bool) -> None:
        """在坐标轴上绘制轨迹
        
        绘制策略：绘制全部轨迹点（让matplotlib自动裁剪超出视口的部分），
        仅在计算方向箭头时提取视口内的连续段，确保轨迹连续不断裂。
        
        Args:
            ax: matplotlib坐标轴
            overlay: 轨迹叠加配置
            extent: 图像范围 [x_min, x_max, y_min, y_max]
            is_local: 是否为局部图（局部图添加方向箭头）
        """
        x_min, x_max, y_min, y_max = extent
        
        if overlay.show_actual and overlay.actual_xy is not None and len(overlay.actual_xy) > 1:
            ax.plot(overlay.actual_xy[:, 0], overlay.actual_xy[:, 1], '-',
                   color=overlay.actual_color,
                   linewidth=overlay.actual_linewidth,
                   label='实际轨迹', zorder=20, alpha=0.9,
                   clip_on=True)
        
        if overlay.show_expected and overlay.expected_xy is not None and len(overlay.expected_xy) > 1:
            ax.plot(overlay.expected_xy[:, 0], overlay.expected_xy[:, 1], '--',
                   color=overlay.expected_color,
                   linewidth=overlay.expected_linewidth,
                   label='期望轨迹', zorder=20, alpha=0.9,
                   clip_on=True)
            
            if is_local:
                vis_segments = self._extract_visible_segments(
                    overlay.expected_xy, x_min, x_max, y_min, y_max)
                for seg in vis_segments:
                    if len(seg) > 1:
                        self._draw_direction_arrows(ax, seg, overlay, extent)
    
    def _extract_visible_segments(self, traj: np.ndarray,
                                  x_min: float, x_max: float,
                                  y_min: float, y_max: float) -> List[np.ndarray]:
        """从轨迹中提取视口内的连续段
        
        将轨迹按是否在视口内分段，返回所有连续可见段。
        这避免了布尔掩码过滤后非连续点导致方向计算错误的问题。
        
        Args:
            traj: 完整轨迹坐标 (N, 2)
            x_min, x_max, y_min, y_max: 视口范围
            
        Returns:
            连续可见段列表，每个元素为 (M, 2) 数组
        """
        margin_x = (x_max - x_min) * 0.1
        margin_y = (y_max - y_min) * 0.1
        in_view = (
            (traj[:, 0] >= x_min - margin_x) & (traj[:, 0] <= x_max + margin_x) &
            (traj[:, 1] >= y_min - margin_y) & (traj[:, 1] <= y_max + margin_y)
        )
        
        segments = []
        current_seg = []
        for i, is_vis in enumerate(in_view):
            if is_vis:
                current_seg.append(traj[i])
            else:
                if len(current_seg) >= 2:
                    segments.append(np.array(current_seg))
                current_seg = []
        if len(current_seg) >= 2:
            segments.append(np.array(current_seg))
        
        return segments
    
    def _draw_direction_arrows(self, ax, vis_pts: np.ndarray,
                               overlay: 'TrajectoryOverlay',
                               extent: list) -> None:
        """在期望轨迹上绘制方向指示箭头
        
        使用FancyArrowPatch绘制箭头，箭头长度从像素转换为数据坐标，
        确保在不同缩放级别下箭头大小一致。
        
        Args:
            ax: matplotlib坐标轴
            vis_pts: 连续可见的期望轨迹点
            overlay: 轨迹叠加配置
            extent: 图像范围 [x_min, x_max, y_min, y_max]
        """
        from matplotlib.patches import FancyArrowPatch
        
        if len(vis_pts) < 2:
            return
        
        fig = ax.get_figure()
        bbox = ax.get_window_extent(fig.canvas.get_renderer() if fig.canvas and hasattr(fig.canvas, 'get_renderer') else None)
        
        x_min, x_max, y_min, y_max = extent
        data_width = x_max - x_min
        data_height = y_max - y_min
        
        if data_width <= 0 or data_height <= 0:
            return
        
        try:
            dpi = fig.dpi
            fig_w_inches = fig.get_figwidth()
            ax_bbox = ax.get_position()
            ax_w_inches = fig_w_inches * ax_bbox.width
            pixels_per_meter = ax_w_inches * dpi / data_width
            arrow_len_data = overlay.arrow_length / pixels_per_meter
        except Exception:
            arrow_len_data = data_width * 0.04
        
        arrow_len_data = max(arrow_len_data, data_width * 0.02)
        arrow_len_data = min(arrow_len_data, data_width * 0.08)
        
        diffs = np.diff(vis_pts, axis=0)
        seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
        cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]
        
        if total_len < overlay.arrow_interval * 0.5:
            mid = len(vis_pts) // 2
            if mid > 0:
                dx = vis_pts[mid, 0] - vis_pts[mid-1, 0]
                dy = vis_pts[mid, 1] - vis_pts[mid-1, 1]
                norm = np.sqrt(dx*dx + dy*dy)
                if norm > 1e-6:
                    dx /= norm; dy /= norm
                    cx = vis_pts[mid, 0]
                    cy = vis_pts[mid, 1]
                    tail = (cx - dx * arrow_len_data / 2, cy - dy * arrow_len_data / 2)
                    tip = (cx + dx * arrow_len_data / 2, cy + dy * arrow_len_data / 2)
                    arrow = FancyArrowPatch(
                        tail, tip,
                        arrowstyle='->', mutation_scale=15,
                        color=overlay.expected_color, linewidth=2,
                        zorder=25, clip_on=True
                    )
                    ax.add_patch(arrow)
            return
        
        n_arrows = max(1, int(total_len / overlay.arrow_interval))
        for i in range(n_arrows):
            target_dist = (i + 0.5) * overlay.arrow_interval
            if target_dist > total_len:
                break
            
            idx = np.searchsorted(cum_len, target_dist) - 1
            idx = max(0, min(idx, len(vis_pts) - 2))
            
            frac = (target_dist - cum_len[idx]) / max(seg_lens[idx], 1e-6)
            frac = np.clip(frac, 0, 1)
            
            px = vis_pts[idx, 0] + frac * diffs[idx, 0]
            py = vis_pts[idx, 1] + frac * diffs[idx, 1]
            
            dx = diffs[idx, 0]
            dy = diffs[idx, 1]
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 1e-6:
                dx /= norm; dy /= norm
                tail = (px - dx * arrow_len_data / 2, py - dy * arrow_len_data / 2)
                tip = (px + dx * arrow_len_data / 2, py + dy * arrow_len_data / 2)
                arrow = FancyArrowPatch(
                    tail, tip,
                    arrowstyle='->', mutation_scale=15,
                    color=overlay.expected_color, linewidth=2,
                    zorder=25, clip_on=True
                )
                ax.add_patch(arrow)
    
    def visualize_dual(self, global_ogm: 'OccupancyGridMap',
                       title: str = '占据网格图 - 全局/局部对比',
                       trajectory_overlay: Optional['TrajectoryOverlay'] = None) -> 'Figure':
        """并列可视化全局和局部占据网格图
        
        Args:
            global_ogm: 全局占据网格图对象
            title: 图表标题
            
        Returns:
            Figure: matplotlib Figure对象
        """
        from matplotlib.figure import Figure
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Rectangle
        
        fig = Figure(figsize=(16, 8), dpi=100)
        
        colors = ['#D0D0D0', '#FFFFFF', '#1A1A1A']
        cmap = ListedColormap(colors)
        
        # --- 全局图 ---
        ax1 = fig.add_subplot(121)
        if global_ogm.grid is not None:
            dg = global_ogm.grid.copy().astype(float)
            dg[dg == global_ogm.config.unknown_value] = 0
            dg[dg == global_ogm.config.free_value] = 1
            dg[dg == global_ogm.config.occupied_value] = 2
            
            g_extent = [
                global_ogm.origin[0],
                global_ogm.origin[0] + global_ogm.grid_size[1] * global_ogm.config.resolution,
                global_ogm.origin[1],
                global_ogm.origin[1] + global_ogm.grid_size[0] * global_ogm.config.resolution,
            ]
            ax1.imshow(dg, cmap=cmap, origin='lower', extent=g_extent,
                      interpolation='nearest', vmin=0, vmax=2)
            
            local_bounds = self.local_bounds
            if local_bounds is not None:
                min_xy, max_xy = local_bounds
                rect_w = max_xy[0] - min_xy[0]
                rect_h = max_xy[1] - min_xy[1]
                rect_fill = Rectangle(
                    (min_xy[0], min_xy[1]), rect_w, rect_h,
                    linewidth=2.5, edgecolor='#FF4444', facecolor='#FF4444',
                    alpha=0.2, linestyle='--', zorder=10
                )
                ax1.add_patch(rect_fill)
                rect_border = Rectangle(
                    (min_xy[0], min_xy[1]), rect_w, rect_h,
                    linewidth=2.5, edgecolor='#FF4444', facecolor='none',
                    linestyle='--', zorder=11
                )
                ax1.add_patch(rect_border)
                cx = (min_xy[0] + max_xy[0]) / 2
                cy = (min_xy[1] + max_xy[1]) / 2
                ax1.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2.5, zorder=12)
                ax1.annotate('局部区域', xy=(min_xy[0], max_xy[1]),
                           fontsize=9, color='#FF4444', fontweight='bold',
                           xytext=(5, 5), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    alpha=0.8, edgecolor='#FF4444'))
            
            if self._local_center is not None:
                ax1.plot(self._local_center[0], self._local_center[1], 'r^',
                        markersize=12, markeredgecolor='darkred', markeredgewidth=1.5,
                        zorder=15, label='无人机位置')
            
            if trajectory_overlay is not None and trajectory_overlay.show_all:
                self._draw_trajectories(ax1, trajectory_overlay, g_extent, is_local=False)
            
            handles1, labels1 = ax1.get_legend_handles_labels()
            if handles1:
                ax1.legend(loc='upper right', fontsize=9)
            
            g_stats = global_ogm.get_statistics()
            g_info = (
                f"网格: {g_stats.get('grid_width', 0)}x{g_stats.get('grid_height', 0)} | "
                f"占据: {g_stats.get('occupancy_rate', 0):.1f}% | "
                f"空闲: {g_stats.get('free_rate', 0):.1f}%"
            )
            ax1.text(0.02, 0.98, g_info, transform=ax1.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax1.set_xlabel('X (m)', fontsize=10)
        ax1.set_ylabel('Y (m)', fontsize=10)
        ax1.set_title('全局占据网格图', fontsize=12, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.set_xlim(g_extent[0], g_extent[1])
        ax1.set_ylim(g_extent[2], g_extent[3])
        ax1.grid(False)
        
        # --- 局部图 ---
        ax2 = fig.add_subplot(122)
        if self.grid is not None:
            dg = self.grid.copy().astype(float)
            dg[dg == self.config.unknown_value] = 0
            dg[dg == self.config.free_value] = 1
            dg[dg == self.config.occupied_value] = 2
            
            l_extent = [
                self.origin[0],
                self.origin[0] + self.grid_size[1] * self.config.resolution,
                self.origin[1],
                self.origin[1] + self.grid_size[0] * self.config.resolution,
            ]
            ax2.imshow(dg, cmap=cmap, origin='lower', extent=l_extent,
                      interpolation='nearest', vmin=0, vmax=2)
            
            if self._local_center is not None:
                ax2.plot(self._local_center[0], self._local_center[1], 'r^',
                        markersize=14, markeredgecolor='darkred', markeredgewidth=2,
                        zorder=15, label='无人机位置')
            
            if trajectory_overlay is not None and trajectory_overlay.show_all:
                self._draw_trajectories(ax2, trajectory_overlay, l_extent, is_local=True)
            
            handles2, labels2 = ax2.get_legend_handles_labels()
            if handles2:
                ax2.legend(loc='upper right', fontsize=9)
            
            l_stats = self.get_statistics()
            l_info = (
                f"网格: {l_stats.get('grid_width', 0)}x{l_stats.get('grid_height', 0)} | "
                f"占据: {l_stats.get('occupancy_rate', 0):.1f}% | "
                f"空闲: {l_stats.get('free_rate', 0):.1f}%"
            )
            ax2.text(0.02, 0.98, l_info, transform=ax2.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        ax2.set_title(f'局部占据网格图 (范围: {self.config.local_range:.1f}m)', fontsize=12, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.set_xlim(l_extent[0], l_extent[1])
        ax2.set_ylim(l_extent[2], l_extent[3])
        ax2.grid(False)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
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
