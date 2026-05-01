# -*- coding: utf-8 -*-
"""
ROS Bag数据可视化模块
功能：绘制位姿数据的各种曲线图，支持多话题对比
作者：Auto-generated
日期：2026-04-27
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    """数据可视化器主类
    
    提供完整的ROS位姿数据可视化功能，包括：
    - 位置数据(x,y,z)随时间变化的曲线
    - 姿态数据(roll,pitch,yaw)随时间变化的曲线
    - 多话题数据对比显示
    - 2D/3D轨迹可视化
    - 滤波前后数据对比
    
    使用示例：
        >>> visualizer = DataVisualizer()
        >>> visualizer.add_pose_data('/Odometry', pose_data)
        >>> visualizer.plot_position_time()  # 绘制位置-时间曲线
        >>> visualizer.show_all_plots()     # 显示所有图表
    
    注意事项：
        - 需要matplotlib库
        - 支持中文标签显示
        - 图表可保存为多种格式(PNG, PDF, SVG)
    """

    def __init__(self):
        """初始化可视化器
        
        创建图形画布和数据存储结构
        """
        self.pose_data_dict: Dict[str, Any] = {}
        self.filtered_data_dict: Dict[str, Any] = {}
        self.figures: Dict[str, plt.Figure] = {}
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
        self.line_styles = ['-', '--', '-.', ':']

    def add_pose_data(self, topic_name: str, pose_data: Any, data_type: str = 'raw') -> None:
        """添加位姿数据到可视化器
        
        Args:
            topic_name (str): 话题名称，用作标识符
            pose_data: PoseData对象或包含timestamp,x,y,z等字段的对象
            data_type (str): 数据类型，'raw'为原始数据，'filtered'为滤波后数据
            
        示例：
            >>> visualizer.add_pose_data('/Odometry', raw_pose_data, 'raw')
            >>> visualizer.add_pose_data('/Odometry', filtered_pose_data, 'filtered')
        """
        if data_type == 'raw':
            self.pose_data_dict[topic_name] = pose_data
        elif data_type == 'filtered':
            self.filtered_data_dict[topic_name] = pose_data
        
        print(f"已添加{('原始' if data_type == 'raw' else '滤波后')}数据: {topic_name}")

    def clear_all_data(self) -> None:
        """清除所有已添加的数据"""
        self.pose_data_dict.clear()
        self.filtered_data_dict.clear()
        self.figures.clear()
        print("已清除所有数据")

    def plot_position_time(self, 
                          topics: Optional[List[str]] = None,
                          show_filtered: bool = True,
                          save_path: Optional[str] = None,
                          figsize: Tuple[float, float] = (14, 10)) -> plt.Figure:
        """绘制位置数据(x,y,z)随时间变化的曲线
        
        创建一个包含3个子图的图表，分别显示X、Y、Z坐标随时间的变化。
        
        Args:
            topics (Optional[List[str]]): 要绘制的话题列表，None表示全部
            show_filtered (bool): 是否同时显示滤波后的数据
            save_path (Optional[str]): 图片保存路径，None则不保存
            figsize (Tuple[float, float]): 图表尺寸(宽,高)，单位为英寸
            
        Returns:
            plt.Figure: matplotlib的Figure对象
            
        图表特性：
            - 包含标题、坐标轴标签、图例和网格线
            - 原始数据显示为实线，滤波数据显示为虚线
            - 自动调整时间轴单位（秒）
            
        示例：
            >>> fig = visualizer.plot_position_time(['/Odometry'], save_path='position.png')
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        fig.suptitle('位置数据 (Position Data) - 时间变化曲线', fontsize=14, fontweight='bold')
        
        # 确定要绘制的话题
        topics_to_plot = topics if topics else list(self.pose_data_dict.keys())
        
        for idx, topic_name in enumerate(topics_to_plot):
            color_idx = idx % len(self.colors)
            color = self.colors[color_idx]
            
            # 绘制原始数据
            if topic_name in self.pose_data_dict:
                data = self.pose_data_dict[topic_name]
                
                if hasattr(data, 'timestamp') and len(data.timestamp) > 0:
                    time_sec = data.timestamp - data.timestamp[0]  # 相对时间
                    
                    axes[0].plot(time_sec, data.x, color=color, linestyle='-',
                               label=f'{topic_name} (原始)', linewidth=1.5, alpha=0.8)
                    axes[1].plot(time_sec, data.y, color=color, linestyle='-',
                               label=f'{topic_name} (原始)', linewidth=1.5, alpha=0.8)
                    axes[2].plot(time_sec, data.z, color=color, linestyle='-',
                               label=f'{topic_name} (原始)', linewidth=1.5, alpha=0.8)
                    
                    # 绘制滤波后的数据
                    if show_filtered and topic_name in self.filtered_data_dict:
                        filtered = self.filtered_data_dict[topic_name]
                        if hasattr(filtered, 'timestamp') and len(filtered.timestamp) > 0:
                            filt_time = filtered.timestamp - filtered.timestamp[0]
                            
                            axes[0].plot(filt_time, filtered.x, color=color, linestyle='--',
                                       label=f'{topic_name} (滤波)', linewidth=1.2, alpha=0.6)
                            axes[1].plot(filt_time, filtered.y, color=color, linestyle='--',
                                       label=f'{topic_name} (滤波)', linewidth=1.2, alpha=0.6)
                            axes[2].plot(filt_time, filtered.z, color=color, linestyle='--',
                                       label=f'{topic_name} (滤波)', linewidth=1.2, alpha=0.6)

        # 设置子图属性
        axis_labels = ['X 坐标 (m)', 'Y 坐标 (m)', 'Z 坐标 (m)']
        for i, ax in enumerate(axes):
            ax.set_ylabel(axis_labels[i], fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlabel('时间 (s)' if i == 2 else '', fontsize=11)
        
        axes[-1].set_xlabel('时间 (s)', fontsize=11)
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"位置曲线图已保存至: {save_path}")
        
        self.figures['position_time'] = fig
        return fig

    def plot_orientation_time(self,
                             topics: Optional[List[str]] = None,
                             show_filtered: bool = True,
                             save_path: Optional[str] = None,
                             figsize: Tuple[float, float] = (14, 10)) -> plt.Figure:
        """绘制姿态数据(roll,pitch,yaw)随时间变化的曲线
        
        创建一个包含3个子图的图表，分别显示翻滚角、俯仰角、偏航角随时间的变化。
        所有角度以度为单位显示。
        
        Args:
            topics (Optional[List[str]]): 要绘制的话题列表
            show_filtered (bool): 是否同时显示滤波后的数据
            save_path (Optional[str]): 图片保存路径
            figsize (Tuple[float, float]): 图表尺寸
            
        Returns:
            plt.Figure: matplotlib的Figure对象
            
        说明：
            - Roll: 绕X轴旋转角度（翻滚）
            - Pitch: 绕Y轴旋转角度（俯仰）
            - Yaw: 绕Z轴旋转角度（偏航）
            - 角度从弧度自动转换为度数显示
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        fig.suptitle('姿态数据 (Orientation Data) - 时间变化曲线', fontsize=14, fontweight='bold')
        
        topics_to_plot = topics if topics else list(self.pose_data_dict.keys())
        
        for idx, topic_name in enumerate(topics_to_plot):
            color_idx = idx % len(self.colors)
            color = self.colors[color_idx]
            
            if topic_name in self.pose_data_dict:
                data = self.pose_data_dict[topic_name]
                
                if hasattr(data, 'timestamp') and len(data.timestamp) > 0:
                    time_sec = data.timestamp - data.timestamp[0]
                    
                    # 弧度转换为度
                    roll_deg = np.degrees(data.roll)
                    pitch_deg = np.degrees(data.pitch)
                    yaw_deg = np.degrees(data.yaw)
                    
                    axes[0].plot(time_sec, roll_deg, color=color, linestyle='-',
                               label=f'{topic_name} (原始)', linewidth=1.5, alpha=0.8)
                    axes[1].plot(time_sec, pitch_deg, color=color, linestyle='-',
                               label=f'{topic_name} (原始)', linewidth=1.5, alpha=0.8)
                    axes[2].plot(time_sec, yaw_deg, color=color, linestyle='-',
                               label=f'{topic_name} (原始)', linewidth=1.5, alpha=0.8)
                    
                    # 滤波后数据
                    if show_filtered and topic_name in self.filtered_data_dict:
                        filtered = self.filtered_data_dict[topic_name]
                        if hasattr(filtered, 'timestamp') and len(filtered.timestamp) > 0:
                            filt_time = filtered.timestamp - filtered.timestamp[0]
                            
                            axes[0].plot(filt_time, np.degrees(filtered.roll), color=color,
                                       linestyle='--', label=f'{topic_name} (滤波)',
                                       linewidth=1.2, alpha=0.6)
                            axes[1].plot(filt_time, np.degrees(filtered.pitch), color=color,
                                       linestyle='--', label=f'{topic_name} (滤波)',
                                       linewidth=1.2, alpha=0.6)
                            axes[2].plot(filt_time, np.degrees(filtered.yaw), color=color,
                                       linestyle='--', label=f'{topic_name} (滤波)',
                                       linewidth=1.2, alpha=0.6)

        axis_labels = ['Roll 角 (°)', 'Pitch 角 (°)', 'Yaw 角 (°)']
        for i, ax in enumerate(axes):
            ax.set_ylabel(axis_labels[i], fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlabel('时间 (s)' if i == 2 else '', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"姿态曲线图已保存至: {save_path}")
        
        self.figures['orientation_time'] = fig
        return fig

    def plot_2d_trajectory(self,
                          topics: Optional[List[str]] = None,
                          show_filtered: bool = True,
                          save_path: Optional[str] = None,
                          figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
        """绘制2D轨迹图(X-Y平面投影)
        
        在二维平面上显示机器人的运动轨迹。
        
        Args:
            topics (Optional[List[str]]): 要绘制的话题列表
            show_filtered (bool): 是否同时显示滤波后的轨迹
            save_path (Optional[str]): 图片保存路径
            figsize (Tuple[float, float]): 图表尺寸
            
        Returns:
            plt.Figure: matplotlib的Figure对象
            
        特性：
            - X轴和Y轴保持相同比例（等比例缩放）
            - 显示起点和终点标记
            - 包含网格线和坐标轴标签
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('2D运动轨迹 (2D Trajectory) - XY平面', fontsize=14, fontweight='bold')
        
        topics_to_plot = topics if topics else list(self.pose_data_dict.keys())
        
        for idx, topic_name in enumerate(topics_to_plot):
            color_idx = idx % len(self.colors)
            color = self.colors[color_idx]
            
            if topic_name in self.pose_data_dict:
                data = self.pose_data_dict[topic_name]
                
                if hasattr(data, 'x') and len(data.x) > 0:
                    # 绘制原始轨迹
                    ax.plot(data.x, data.y, color=color, linestyle='-',
                           label=f'{topic_name} (原始)', linewidth=1.5, alpha=0.8)
                    
                    # 标记起点和终点
                    ax.scatter(data.x[0], data.y[0], color=color, marker='o',
                              s=100, zorder=5, edgecolors='black', linewidths=1.5)
                    ax.scatter(data.x[-1], data.y[-1], color=color, marker='s',
                              s=100, zorder=5, edgecolors='black', linewidths=1.5)
                    
                    # 绘制滤波后轨迹
                    if show_filtered and topic_name in self.filtered_data_dict:
                        filtered = self.filtered_data_dict[topic_name]
                        if hasattr(filtered, 'x') and len(filtered.x) > 0:
                            ax.plot(filtered.x, filtered.y, color=color, linestyle='--',
                                   label=f'{topic_name} (滤波)', linewidth=1.2, alpha=0.6)

        ax.set_xlabel('X 坐标 (m)', fontsize=12)
        ax.set_ylabel('Y 坐标 (m)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', fontsize=10)
        ax.axis('equal')  # 保持比例一致
        ax.set_aspect('equal', adjustable='box')
        
        # 添加起点终点说明
        ax.annotate('起点', xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=9, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.annotate('终点', xy=(0.02, 0.93), xycoords='axes fraction',
                   fontsize=9, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D轨迹图已保存至: {save_path}")
        
        self.figures['trajectory_2d'] = fig
        return fig

    def plot_3d_trajectory(self,
                          topics: Optional[List[str]] = None,
                          show_filtered: bool = True,
                          save_path: Optional[str] = None,
                          figsize: Tuple[float, float] = (12, 9)) -> plt.Figure:
        """绘制3D轨迹图
        
        在三维空间中显示完整的运动轨迹。
        
        Args:
            topics (Optional[List[str]]): 要绘制的话题列表
            show_filtered (bool): 是否同时显示滤波后的轨迹
            save_path (Optional[str]): 图片保存路径
            figsize (Tuple[float, float]): 图表尺寸
            
        Returns:
            plt.Figure: matplotlib的Figure对象
            
        特性：
            - 支持3D交互式查看（在支持的界面中）
            - 可旋转视角观察轨迹形状
            - 显示坐标轴刻度和标签
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('3D运动轨迹 (3D Trajectory)', fontsize=14, fontweight='bold')
        
        topics_to_plot = topics if topics else list(self.pose_data_dict.keys())
        
        for idx, topic_name in enumerate(topics_to_plot):
            color_idx = idx % len(self.colors)
            color = self.colors[color_idx]
            
            if topic_name in self.pose_data_dict:
                data = self.pose_data_dict[topic_name]
                
                if hasattr(data, 'x') and len(data.x) > 0:
                    # 绘制原始3D轨迹
                    ax.plot(data.x, data.y, data.z, color=color, linestyle='-',
                           label=f'{topic_name} (原始)', linewidth=1.5, alpha=0.8)
                    
                    # 标记起点和终点
                    ax.scatter([data.x[0]], [data.y[0]], [data.z[0]],
                              color=color, marker='o', s=100, zorder=5)
                    ax.scatter([data.x[-1]], [data.y[-1]], [data.z[-1]],
                              color=color, marker='s', s=100, zorder=5)
                    
                    # 滤波后轨迹
                    if show_filtered and topic_name in self.filtered_data_dict:
                        filtered = self.filtered_data_dict[topic_name]
                        if hasattr(filtered, 'x') and len(filtered.x) > 0:
                            ax.plot(filtered.x, filtered.y, filtered.z, color=color,
                                   linestyle='--', label=f'{topic_name} (滤波)',
                                   linewidth=1.2, alpha=0.6)

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        
        # 调整视角
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D轨迹图已保存至: {save_path}")
        
        self.figures['trajectory_3d'] = fig
        return fig

    def plot_comparison(self,
                       topic_name: str,
                       data_field: str = 'x',
                       save_path: Optional[str] = None,
                       figsize: Tuple[float, float] = (12, 6)) -> plt.Figure:
        """绘制单个字段的滤波前后对比图
        
        用于详细对比某个特定数据字段在滤波前后的差异，
        便于评估滤波效果。
        
        Args:
            topic_name (str): 话题名称
            data_field (str): 数据字段名 ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
            save_path (Optional[str]): 图片保存路径
            figsize (Tuple[float, float]): 图表尺寸
            
        Returns:
            plt.Figure: matplotlib的Figure对象
            
        图表内容：
            - 上半部分：原始数据和滤波后数据的叠加对比
            - 下半部分：两者之间的残差（误差）
            
        示例：
            >>> fig = visualizer.plot_comparison('/Odometry', 'x', 'comparison_x.png')
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        field_labels = {
            'x': 'X 坐标 (m)',
            'y': 'Y 坐标 (m)',
            'z': 'Z 坐标 (m)',
            'roll': 'Roll 角 (°)',
            'pitch': 'Pitch 角 (°)',
            'yaw': 'Yaw 角 (°)'
        }
        
        field_unit = '(°)' if data_field in ['roll', 'pitch', 'yaw'] else '(m)'
        title = f"'{topic_name}' - {data_field.upper()} 数据对比"
        fig.suptitle(title, fontsize=13, fontweight='bold')
        
        if topic_name in self.pose_data_dict and topic_name in self.filtered_data_dict:
            raw_data = self.pose_data_dict[topic_name]
            filt_data = self.filtered_data_dict[topic_name]
            
            if (hasattr(raw_data, data_field) and hasattr(filt_data, data_field) and
                len(raw_data.timestamp) > 0 and len(filt_data.timestamp) > 0):
                
                # 获取数据
                raw_values = getattr(raw_data, data_field)
                filt_values = getattr(filt_data, data_field)
                
                # 如果是角度，转换为度数
                if data_field in ['roll', 'pitch', 'yaw']:
                    raw_values = np.degrees(raw_values)
                    filt_values = np.degrees(filt_values)
                
                # 时间轴（使用较短的数据长度）
                min_len = min(len(raw_data.timestamp), len(filt_data.timestamp))
                time_raw = raw_data.timestamp[:min_len] - raw_data.timestamp[0]
                time_filt = filt_data.timestamp[:min_len] - filt_data.timestamp[0]
                
                # 上半部分：数据对比
                ax1.plot(time_raw, raw_values[:min_len], 'b-', label='原始数据',
                        linewidth=1.5, alpha=0.8)
                ax1.plot(time_filt, filt_values[:min_len], 'r-', label='滤波后数据',
                        linewidth=1.5, alpha=0.8)
                ax1.set_ylabel(field_labels.get(data_field, data_field), fontsize=11)
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend(loc='upper right', fontsize=10)
                ax1.set_title('数据对比', fontsize=11)
                
                # 下半部分：残差
                residual = raw_values[:min_len] - filt_values[:min_len]
                ax2.plot(time_raw, residual, 'g-', linewidth=1.0, alpha=0.8)
                ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
                ax2.fill_between(time_raw, residual, 0, alpha=0.3, color='green')
                ax2.set_xlabel('时间 (s)', fontsize=11)
                ax2.set_ylabel(f'残差 {field_unit}', fontsize=11)
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.set_title(f'残差 (最大: {np.max(np.abs(residual)):.4f})', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存至: {save_path}")
        
        self.figures[f'comparison_{topic_name}_{data_field}'] = fig
        return fig

    def create_dashboard(self,
                        topics: Optional[List[str]] = None,
                        save_path: Optional[str] = None,
                        figsize: Tuple[float, float] = (16, 12)) -> plt.Figure:
        """创建综合仪表板视图
        
        将多个关键图表组合到一个大图中，提供全面的数据概览。
        
        Args:
            topics (Optional[List[str]]): 要展示的话题列表
            save_path (Optional[str]): 图片保存路径
            figsize (Tuple[float, float]): 总图表尺寸
            
        Returns:
            plt.Figure: matplotlib的Figure对象
            
        仪表板布局：
            - 左上：位置-时间曲线 (X,Y,Z)
            - 右上：姿态-时间曲线 (Roll,Pitch,Yaw)
            - 左下：2D轨迹图 (XY平面)
            - 右下：速度/统计信息
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle('ROS Bag数据分析仪表板 (Data Dashboard)', 
                    fontsize=16, fontweight='bold')
        
        # 创建子图布局
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        ax_pos = fig.add_subplot(gs[0, 0])
        ax_ori = fig.add_subplot(gs[0, 1])
        ax_traj = fig.add_subplot(gs[1, 0])
        ax_stats = fig.add_subplot(gs[1, 1])
        
        topics_to_plot = topics if topics else list(self.pose_data_dict.keys())[:3]  # 最多3个话题
        
        for idx, topic_name in enumerate(topics_to_plot[:2]):  # 限制数量避免过于拥挤
            color = self.colors[idx % len(self.colors)]
            
            if topic_name in self.pose_data_dict:
                data = self.pose_data_dict[topic_name]
                
                if hasattr(data, 'timestamp') and len(data.timestamp) > 0:
                    t = data.timestamp - data.timestamp[0]
                    
                    # 位置曲线（只显示X和Y以节省空间）
                    ax_pos.plot(t, data.x, color=color, linestyle='-',
                               label=f'{topic_name}_X', linewidth=1.2, alpha=0.8)
                    ax_pos.plot(t, data.y, color=color, linestyle='--',
                               label=f'{topic_name}_Y', linewidth=1.2, alpha=0.8)
                    
                    # 姿态曲线（只显示Yaw）
                    ax_ori.plot(t, np.degrees(data.yaw), color=color, linestyle='-',
                               label=f'{topic_name}', linewidth=1.2, alpha=0.8)
                    
                    # 2D轨迹
                    ax_traj.plot(data.x, data.y, color=color, linestyle='-',
                                label=topic_name, linewidth=1.5, alpha=0.8)
        
        # 配置各子图
        ax_pos.set_title('位置数据 (Position)', fontsize=11, fontweight='bold')
        ax_pos.set_xlabel('时间 (s)')
        ax_pos.set_ylabel('坐标 (m)')
        ax_pos.grid(True, linestyle='--', alpha=0.6)
        ax_pos.legend(loc='upper right', fontsize=8)
        
        ax_ori.set_title('偏航角 (Yaw Angle)', fontsize=11, fontweight='bold')
        ax_ori.set_xlabel('时间 (s)')
        ax_ori.set_ylabel('角度 (°)')
        ax_ori.grid(True, linestyle='--', alpha=0.6)
        ax_ori.legend(loc='upper right', fontsize=8)
        
        ax_traj.set_title('2D轨迹 (XY Plane)', fontsize=11, fontweight='bold')
        ax_traj.set_xlabel('X (m)')
        ax_traj.set_ylabel('Y (m)')
        ax_traj.grid(True, linestyle='--', alpha=0.6)
        ax_traj.axis('equal')
        ax_traj.legend(loc='best', fontsize=8)
        
        # 统计信息文本框
        ax_stats.axis('off')
        stats_text = "数据统计信息\n" + "=" * 30 + "\n\n"
        
        for topic_name in topics_to_plot:
            if topic_name in self.pose_data_dict:
                data = self.pose_data_dict[topic_name]
                if hasattr(data, 'timestamp') and len(data.timestamp) > 0:
                    stats_text += f"话题: {topic_name}\n"
                    stats_text += f"  数据点数: {len(data.timestamp)}\n"
                    stats_text += f"  X范围: [{data.x.min():.3f}, {data.x.max():.3f}] m\n"
                    stats_text += f"  Y范围: [{data.y.min():.3f}, {data.y.max():.3f}] m\n"
                    stats_text += f"  持续时间: {(data.timestamp[-1]-data.timestamp[0]):.2f} s\n\n"
        
        ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_stats.set_title('统计信息 (Statistics)', fontsize=11, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"仪表板已保存至: {save_path}")
        
        self.figures['dashboard'] = fig
        return fig

    def show_all_plots(self) -> None:
        """显示所有已创建的图表
        
        使用matplotlib的交互式窗口显示所有图表。
        每个图表在单独的窗口中显示。
        """
        print("正在显示所有图表...")
        for name, fig in self.figures.items():
            plt.figure(fig.number)
            print(f"  显示图表: {name}")
        plt.show()

    def save_all_plots(self, output_dir: str = './plots',
                      format: str = 'png',
                      dpi: int = 300) -> Dict[str, str]:
        """保存所有图表到指定目录
        
        Args:
            output_dir (str): 输出目录路径
            format (str): 图片格式 ('png', 'pdf', 'svg', 'jpg')
            dpi (int): 图片分辨率
            
        Returns:
            Dict[str, str]: 以图表名称为键、文件路径为值的字典
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = {}
        for name, fig in self.figures.items():
            file_path = os.path.join(output_dir, f'{name}.{format}')
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight',
                       format=format)
            saved_files[name] = file_path
            print(f"已保存: {file_path}")
        
        print(f"\n共保存 {len(saved_files)} 个图表至: {output_dir}")
        return saved_files

    def close_all_figures(self) -> None:
        """关闭所有打开的图表，释放内存"""
        import matplotlib.pyplot as plt
        plt.close('all')
        self.figures.clear()
        print("已关闭所有图表")


def main():
    """测试函数 - DataVisualizer模块的功能演示"""
    print("=" * 60)
    print("数据可视化器测试")
    print("=" * 60)
    
    # 创建模拟数据进行测试
    from bag_parser import PoseData
    
    # 生成模拟位姿数据
    n_points = 100
    t = np.linspace(0, 10, n_points)
    
    test_pose = PoseData(
        timestamp=t,
        x=np.sin(t) + np.random.normal(0, 0.05, n_points),
        y=np.cos(t) + np.random.normal(0, 0.05, n_points),
        z=np.linspace(0, 2, n_points),
        roll=np.random.normal(0, 0.01, n_points),
        pitch=np.random.normal(0, 0.01, n_points),
        yaw=t * 0.5,  # 缓慢增加的偏航角
        quaternion_w=np.ones(n_points),
        quaternion_x=np.zeros(n_points),
        quaternion_y=np.zeros(n_points),
        quaternion_z=np.zeros(n_points)
    )
    
    # 创建可视化器并测试
    visualizer = DataVisualizer()
    visualizer.add_pose_data('/test_odometry', test_pose)
    
    # 绘制各类图表
    print("\n生成测试图表...")
    
    try:
        # 位置-时间曲线
        fig1 = visualizer.plot_position_time(save_path='./plots/test_position.png')
        print("✓ 位置-时间曲线已生成")
        
        # 姿态-时间曲线
        fig2 = visualizer.plot_orientation_time(save_path='./plots/test_orientation.png')
        print("✓ 姿态-时间曲线已生成")
        
        # 2D轨迹
        fig3 = visualizer.plot_2d_trajectory(save_path='./plots/test_trajectory_2d.png')
        print("✓ 2D轨迹图已生成")
        
        # 3D轨迹
        fig4 = visualizer.plot_3d_trajectory(save_path='./plots/test_trajectory_3d.png')
        print("✓ 3D轨迹图已生成")
        
        # 仪表板
        fig5 = visualizer.create_dashboard(save_path='./plots/test_dashboard.png')
        print("✓ 综合仪表板已生成")
        
        print("\n所有测试图表生成完成！")
        print("图表文件保存在 ./plots/ 目录中")
        
    except Exception as e:
        print(f"生成图表时出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
