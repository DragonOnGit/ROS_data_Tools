# -*- coding: utf-8 -*-
"""
八叉树地图生成模块
功能：基于点云数据构建八叉树空间索引，支持参数配置、分层显示和可视化

核心算法：
    八叉树通过递归细分3D空间为8个子立方体（体素），每个节点代表一个空间区域。
    当某区域内的点云密度低于阈值时停止细分，叶节点存储该区域的统计信息。

使用示例：
    from octree_map import OctreeMap, OctreeConfig
    
    config = OctreeConfig(voxel_size=0.1, max_depth=8)
    octree = OctreeMap(config)
    octree.build(points)
    octree.visualize()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import time


@dataclass
class OctreeConfig:
    """八叉树配置参数类
    
    Attributes:
        voxel_size (float): 体素大小（米），控制最小空间分辨率。
                            值越小精度越高但内存占用越大。范围: [0.01, 10.0]
        max_depth (int): 最大递归深度。限制八叉树的最大细分层数。
                         范围: [1, 16]，默认8
        min_depth (int): 最小递归深度。强制细分到指定层数后再判断密度。
                         范围: [0, max_depth]，默认0
        density_threshold (int): 密度阈值。当体素内点数少于此值时停止细分。
                                范围: [1, 1000]，默认5
        height_min (float): 高度过滤下限（米）。仅保留高于此值的点。
                            设为负无穷则不过滤。默认: -inf
        height_max (float): 高度过滤上限（米）。仅保留低于此值的点。
                            可用于剥离顶部冗余结构。默认: inf
        color_scheme (str): 颜色映射方案。可选: 'height', 'density', 'depth', 'custom'
        transparency (float): 整体透明度。范围: [0.0, 1.0]，1.0为不透明
        show_borders (bool): 是否显示体素边界线
        render_quality (str): 渲染质量。可选: 'low', 'medium', 'high'
    """
    voxel_size: float = 0.1
    max_depth: int = 8
    min_depth: int = 0
    density_threshold: int = 5
    height_min: float = float('-inf')
    height_max: float = float('inf')
    color_scheme: str = 'height'
    transparency: float = 0.8
    show_borders: bool = False
    render_quality: str = 'medium'
    
    def __post_init__(self):
        self.voxel_size = np.clip(self.voxel_size, 0.01, 10.0)
        self.max_depth = int(np.clip(self.max_depth, 1, 16))
        self.min_depth = int(np.clip(self.min_depth, 0, self.max_depth))
        self.density_threshold = int(np.clip(self.density_threshold, 1, 1000))
        self.transparency = np.clip(self.transparency, 0.0, 1.0)
        if self.color_scheme not in ('height', 'density', 'depth', 'custom'):
            self.color_scheme = 'height'
        if self.render_quality not in ('low', 'medium', 'high'):
            self.render_quality = 'medium'


@dataclass
class OctreeNode:
    """八叉树节点类
    
    Attributes:
        center (np.ndarray): 节点中心坐标 (3,)
        size (float): 节点边长
        depth (int): 节点深度（根节点为0）
        point_count (int): 该节点包含的点数
        point_indices (np.ndarray): 包含的点在原始点云中的索引
        children (List): 8个子节点的列表，None表示未细分
        is_leaf (bool): 是否为叶节点
        centroid (np.ndarray): 包含点的质心坐标 (3,)
    """
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    size: float = 1.0
    depth: int = 0
    point_count: int = 0
    point_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    children: List = field(default_factory=lambda: [None] * 8)
    is_leaf: bool = True
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))


class OctreeMap:
    """八叉树地图生成器
    
    基于点云数据构建八叉树空间索引，支持：
    - 可配置的体素大小和递归深度
    - 基于密度的自适应细分
    - 高度阈值分层显示
    - 多种颜色映射方案
    
    使用示例：
        >>> config = OctreeConfig(voxel_size=0.1, max_depth=8)
        >>> octree = OctreeMap(config)
        >>> octree.build(points)
        >>> leaf_voxels = octree.get_leaf_voxels()
        >>> octree.visualize()
    """
    
    def __init__(self, config: Optional[OctreeConfig] = None):
        """初始化八叉树地图生成器
        
        Args:
            config: 八叉树配置参数，默认为None时使用默认配置
        """
        self.config = config if config is not None else OctreeConfig()
        self.root: Optional[OctreeNode] = None
        self.points: Optional[np.ndarray] = None
        self.filtered_points: Optional[np.ndarray] = None
        self.filtered_indices: Optional[np.ndarray] = None
        self._leaf_voxels: List[Dict] = []
        self._build_time: float = 0.0
        self._total_nodes: int = 0
        self._total_leaves: int = 0
    
    @property
    def build_time(self) -> float:
        """构建耗时（秒）"""
        return self._build_time
    
    @property
    def total_nodes(self) -> int:
        """总节点数"""
        return self._total_nodes
    
    @property
    def total_leaves(self) -> int:
        """总叶节点数"""
        return self._total_leaves
    
    def build(self, points: np.ndarray) -> None:
        """构建八叉树
        
        Args:
            points: 点云数据，形状为(N, 3)或(N, 6)（含颜色RGB）
            
        Raises:
            ValueError: 点云数据为空或格式不正确
        """
        start_time = time.time()
        
        if points is None or len(points) == 0:
            raise ValueError("点云数据为空")
        
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"点云数据格式错误，期望(N,3)或(N,6)，实际{points.shape}")
        
        self.points = points.copy()
        
        # 高度过滤
        z_values = points[:, 2]
        mask = (z_values >= self.config.height_min) & (z_values <= self.config.height_max)
        self.filtered_indices = np.where(mask)[0]
        self.filtered_points = points[mask]
        
        if len(self.filtered_points) == 0:
            print("[WARN] 高度过滤后无剩余点，请调整高度范围")
            self._build_time = time.time() - start_time
            return
        
        # 计算包围盒
        bbox_min = self.filtered_points[:, :3].min(axis=0)
        bbox_max = self.filtered_points[:, :3].max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        size = float((bbox_max - bbox_min).max()) * 1.01  # 略微扩大避免边界问题
        
        # 创建根节点
        self.root = OctreeNode(
            center=center,
            size=size,
            depth=0,
            point_count=len(self.filtered_points),
            point_indices=np.arange(len(self.filtered_points)),
            is_leaf=True,
            centroid=center.copy()
        )
        
        # 递归构建
        self._total_nodes = 1
        self._total_leaves = 0
        self._subdivide(self.root)
        
        # 收集叶节点
        self._leaf_voxels = []
        self._collect_leaves(self.root)
        
        self._build_time = time.time() - start_time
        
        print(f"[OK] 八叉树构建完成:")
        print(f"   原始点数: {len(points)}")
        print(f"   过滤后点数: {len(self.filtered_points)}")
        print(f"   总节点数: {self._total_nodes}")
        print(f"   叶节点数: {self._total_leaves}")
        print(f"   耗时: {self._build_time:.3f}s")
    
    def _subdivide(self, node: OctreeNode) -> None:
        """递归细分八叉树节点
        
        当节点内的点数超过密度阈值且未达到最大深度时，
        将节点细分为8个子节点。
        
        Args:
            node: 待细分的节点
        """
        # 判断是否需要继续细分
        should_subdivide = (
            node.point_count > self.config.density_threshold and
            node.depth < self.config.max_depth
        )
        
        # 强制细分到最小深度
        force_subdivide = node.depth < self.config.min_depth
        
        if not should_subdivide and not force_subdivide:
            node.is_leaf = True
            self._total_leaves += 1
            return
        
        # 检查体素大小是否已经小于最小体素
        child_size = node.size / 2.0
        if child_size < self.config.voxel_size and not force_subdivide:
            node.is_leaf = True
            self._total_leaves += 1
            return
        
        # 计算质心
        if node.point_count > 0:
            node.centroid = self.filtered_points[node.point_indices, :3].mean(axis=0)
        
        # 细分为8个子节点
        node.is_leaf = False
        half = node.size / 4.0
        
        # 8个子节点的偏移量（按二进制编码：x=bit0, y=bit1, z=bit2）
        offsets = np.array([
            [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
            [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1]
        ], dtype=float) * half
        
        # 分配点到子节点
        pts = self.filtered_points[node.point_indices, :3]
        
        for i in range(8):
            child_center = node.center + offsets[i]
            
            # 计算子节点的边界
            child_min = child_center - child_size / 2.0
            child_max = child_center + child_size / 2.0
            
            # 找出属于该子节点的点
            in_child = np.all((pts >= child_min) & (pts < child_max), axis=1)
            child_indices = node.point_indices[in_child]
            
            if len(child_indices) > 0:
                child_node = OctreeNode(
                    center=child_center,
                    size=child_size,
                    depth=node.depth + 1,
                    point_count=len(child_indices),
                    point_indices=child_indices,
                    is_leaf=True,
                    centroid=self.filtered_points[child_indices, :3].mean(axis=0)
                )
                node.children[i] = child_node
                self._total_nodes += 1
                
                # 递归细分
                self._subdivide(child_node)
            else:
                node.children[i] = None
    
    def _collect_leaves(self, node: OctreeNode) -> None:
        """递归收集所有叶节点信息
        
        Args:
            node: 当前节点
        """
        if node is None:
            return
        
        if node.is_leaf and node.point_count > 0:
            centroid = node.centroid
            self._leaf_voxels.append({
                'center': centroid.copy(),
                'size': node.size,
                'depth': node.depth,
                'point_count': node.point_count,
                'point_indices': node.point_indices.copy(),
                'z': centroid[2],
                'density': node.point_count / (node.size ** 3)
            })
            return
        
        for child in node.children:
            if child is not None:
                self._collect_leaves(child)
    
    def get_leaf_voxels(self) -> List[Dict]:
        """获取所有叶节点的体素信息
        
        Returns:
            List[Dict]: 每个元素包含center, size, depth, point_count, z, density等字段
        """
        return self._leaf_voxels
    
    def get_voxel_centers(self) -> np.ndarray:
        """获取所有体素中心坐标
        
        Returns:
            np.ndarray: 形状为(M, 3)的坐标数组
        """
        if not self._leaf_voxels:
            return np.array([]).reshape(0, 3)
        return np.array([v['center'] for v in self._leaf_voxels])
    
    def get_voxel_sizes(self) -> np.ndarray:
        """获取所有体素大小
        
        Returns:
            np.ndarray: 形状为(M,)的体素边长数组
        """
        if not self._leaf_voxels:
            return np.array([])
        return np.array([v['size'] for v in self._leaf_voxels])
    
    def get_statistics(self) -> Dict:
        """获取八叉树统计信息
        
        Returns:
            Dict: 包含各项统计指标的字典
        """
        if not self._leaf_voxels:
            return {}
        
        sizes = np.array([v['size'] for v in self._leaf_voxels])
        counts = np.array([v['point_count'] for v in self._leaf_voxels])
        depths = np.array([v['depth'] for v in self._leaf_voxels])
        densities = np.array([v['density'] for v in self._leaf_voxels])
        
        return {
            'total_points': len(self.filtered_points) if self.filtered_points is not None else 0,
            'filtered_points': len(self.filtered_points) if self.filtered_points is not None else 0,
            'total_nodes': self._total_nodes,
            'total_leaves': self._total_leaves,
            'build_time': self._build_time,
            'voxel_size_min': float(sizes.min()),
            'voxel_size_max': float(sizes.max()),
            'voxel_size_mean': float(sizes.mean()),
            'depth_min': int(depths.min()),
            'depth_max': int(depths.max()),
            'points_per_voxel_min': int(counts.min()),
            'points_per_voxel_max': int(counts.max()),
            'points_per_voxel_mean': float(counts.mean()),
            'density_min': float(densities.min()),
            'density_max': float(densities.max()),
            'density_mean': float(densities.mean()),
        }
    
    def query_point(self, point: np.ndarray) -> Optional[Dict]:
        """查询点所在的体素信息
        
        Args:
            point: 查询点坐标 (3,)
            
        Returns:
            Optional[Dict]: 体素信息字典，未找到返回None
        """
        if self.root is None:
            return None
        
        node = self.root
        while not node.is_leaf:
            # 确定属于哪个子节点
            idx = 0
            if point[0] >= node.center[0]:
                idx |= 1
            if point[1] >= node.center[1]:
                idx |= 2
            if point[2] >= node.center[2]:
                idx |= 4
            
            child = node.children[idx]
            if child is None:
                return None
            node = child
        
        return {
            'center': node.centroid.copy(),
            'size': node.size,
            'depth': node.depth,
            'point_count': node.point_count
        }
    
    def ray_cast(self, origin: np.ndarray, direction: np.ndarray, max_dist: float = 100.0) -> Optional[Dict]:
        """射线投射查询（简化版）
        
        沿射线方向查找第一个非空体素。
        
        Args:
            origin: 射线起点 (3,)
            direction: 射线方向 (3,)，需为单位向量
            max_dist: 最大查询距离
            
        Returns:
            Optional[Dict]: 命中体素信息，未命中返回None
        """
        if self.root is None:
            return None
        
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        step = self.config.voxel_size * 0.5
        t = 0.0
        
        while t < max_dist:
            point = origin + direction * t
            result = self.query_point(point)
            if result is not None and result['point_count'] > 0:
                result['distance'] = t
                return result
            t += step
        
        return None
    
    def export_voxels(self, filepath: str) -> bool:
        """导出体素数据到文件
        
        Args:
            filepath: 输出文件路径（.npz格式）
            
        Returns:
            bool: 导出是否成功
        """
        if not self._leaf_voxels:
            print("[WARN] 无体素数据可导出")
            return False
        
        try:
            centers = np.array([v['center'] for v in self._leaf_voxels])
            sizes = np.array([v['size'] for v in self._leaf_voxels])
            counts = np.array([v['point_count'] for v in self._leaf_voxels])
            depths = np.array([v['depth'] for v in self._leaf_voxels])
            
            np.savez(filepath,
                     centers=centers, sizes=sizes,
                     counts=counts, depths=depths,
                     config={
                         'voxel_size': self.config.voxel_size,
                         'max_depth': self.config.max_depth,
                         'density_threshold': self.config.density_threshold
                     })
            print(f"[OK] 体素数据已导出: {filepath}")
            return True
        except Exception as e:
            print(f"[ERROR] 导出失败: {str(e)}")
            return False
