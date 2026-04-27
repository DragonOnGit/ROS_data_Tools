# -*- coding: utf-8 -*-
"""
数据修改与滤波处理模块
功能：提供多种滤波算法和数据修改接口
作者：Auto-generated
日期：2026-04-27
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import copy


@dataclass
class FilterConfig:
    """滤波器配置类
    
    存储滤波算法的参数配置
    
    Attributes:
        filter_type (str): 滤波器类型 ('moving_average', 'kalman', 'lowpass', 'median')
        window_size (int): 窗口大小（用于滑动平均、中值滤波等）
        alpha (float): 低通滤波系数 (0-1)，值越小平滑效果越强
        process_noise (float): 过程噪声协方差（卡尔曼滤波）
        measurement_noise (float): 测量噪声协方差（卡尔曼滤波）
        estimated_error (float): 初始估计误差协方差（卡尔曼滤波）
        cutoff_freq (float): 截止频率（低通滤波，Hz）
        order (int): 滤波器阶数（巴特沃斯滤波等）
    """
    filter_type: str = 'moving_average'
    window_size: int = 5
    alpha: float = 0.3
    process_noise: float = 0.01
    measurement_noise: float = 0.1
    estimated_error: float = 1.0
    cutoff_freq: float = 5.0
    order: int = 4


class KalmanFilter1D:
    """一维卡尔曼滤波器实现
    
    实现简单的一维卡尔曼滤波，适用于单个数据序列的滤波。
    
    状态空间模型：
        x(k) = x(k-1) + w(k-1)   (状态方程)
        z(k) = x(k) + v(k)       (观测方程)
    
    其中：
        w(k-1): 过程噪声 ~ N(0, Q)
        v(k): 测量噪声 ~ N(0, R)
    
    使用示例：
        >>> kf = KalmanFilter1D(process_noise=0.01, measurement_noise=0.1)
        >>> filtered_data = kf.filter(raw_data)
        
    参数说明：
        - process_noise (Q): 过程噪声协方差，表示系统模型的不确定性
          较小的值意味着更信任模型预测
        - measurement_noise (R): 测量噪声协方差，表示传感器测量的不确定性
          较小的值意味着更信任测量值
    """
    
    def __init__(self, 
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 estimated_error: float = 1.0):
        """初始化卡尔曼滤波器
        
        Args:
            process_noise (float): 过程噪声协方差 Q
            measurement_noise (float): 测量噪声协方差 R
            estimated_error (float): 初始估计误差 P(0)
        """
        self.Q = process_noise      # 过程噪声
        self.R = measurement_noise  # 测量噪声
        self.P = estimated_error    # 估计误差协方差
        self.K = 0.0                # 卡尔曼增益
        self.x = 0.0                # 状态估计值
    
    def reset(self):
        """重置滤波器状态"""
        self.P = 1.0
        self.K = 0.0
        self.x = 0.0
    
    def filter(self, data: np.ndarray) -> np.ndarray:
        """对整个数据序列进行卡尔曼滤波
        
        Args:
            data (np.ndarray): 输入的一维数据数组
            
        Returns:
            np.ndarray: 滤波后的数据数组
            
        算法步骤：
            1. 预测阶段：x_pred = x(k-1), P_pred = P(k-1) + Q
            2. 更新阶段：计算卡尔曼增益K，更新状态和误差
        """
        self.reset()
        n = len(data)
        result = np.zeros(n)
        
        # 初始化：使用第一个测量值作为初始估计
        if n > 0:
            self.x = data[0]
        
        for i in range(n):
            # 预测阶段（对于简单的随机游走模型，预测值等于上一时刻估计）
            x_pred = self.x
            P_pred = self.P + self.Q
            
            # 更新阶段
            self.K = P_pred / (P_pred + self.R)  # 计算卡尔曼增益
            self.x = x_pred + self.K * (data[i] - x_pred)  # 更新状态估计
            self.P = (1 - self.K) * P_pred  # 更新误差协方差
            
            result[i] = self.x
        
        return result


class FilterProcessor:
    """数据滤波处理器主类
    
    提供多种滤波算法的实现，包括：
    - 滑动平均滤波 (Moving Average)
    - 加权滑动平均滤波 (Weighted Moving Average)
    - 中值滤波 (Median Filter)
    - 卡尔曼滤波 (Kalman Filter)
    - 低通滤波 (Low-pass/Butterworth Filter)
    - 指数加权移动平均 (Exponential Weighted MA)
    - Savitzky-Golay滤波
    
    同时支持数据修改功能：
    - 手动调整数据点
    - 剔除异常值
    - 数据插值
    - 数据裁剪与拼接
    
    使用示例：
        >>> processor = FilterProcessor()
        >>> config = FilterConfig(filter_type='moving_average', window_size=5)
        >>> filtered_data = processor.apply_filter(raw_data, config)
        
    注意事项：
        - 不同滤波器适用于不同类型的数据和噪声特征
        - 滤波会引入相位延迟（特别是因果滤波器）
        - 应根据实际需求选择合适的滤波方法和参数
    """

    def __init__(self):
        """初始化滤波处理器"""
        self.available_filters = [
            'moving_average',
            'weighted_moving_average',
            'exponential_moving_average',
            'median',
            'kalman',
            'lowpass_butterworth',
            'savitzky_golay'
        ]
        self.last_config: Optional[FilterConfig] = None
        self.filter_history: Dict[str, Any] = {}

    def apply_filter(self, 
                    data: np.ndarray,
                    config: FilterConfig,
                    return_info: bool = False) -> np.ndarray:
        """应用指定的滤波器对数据进行滤波
        
        根据配置对象中指定的滤波类型调用相应的滤波函数。
        
        Args:
            data (np.ndarray): 待滤波的一维数据数组
            config (FilterConfig): 滤波器配置对象
            return_info (bool): 是否返回额外的滤波信息
            
        Returns:
            np.ndarray: 滤波后的数据数组
            
        Raises:
            ValueError: 当滤波类型不支持时抛出
            
        示例：
            >>> config = FilterConfig(filter_type='kalman', process_noise=0.01)
            >>> filtered = processor.apply_filter(data, config)
            
        可用滤波器：
            - moving_average: 简单滑动平均，适合高频噪声
            - kalman: 卡尔曼滤波，适合高斯白噪声
            - lowpass_butterworth: 巴特沃斯低通，适合特定频率噪声
            - median: 中值滤波，适合脉冲噪声
        """
        if len(data) == 0:
            return data.copy()
        
        self.last_config = config
        
        # 调用对应的滤波方法
        filter_methods = {
            'moving_average': self._moving_average_filter,
            'weighted_moving_average': self._weighted_moving_average_filter,
            'exponential_moving_average': self._exponential_moving_average_filter,
            'median': self._median_filter,
            'kalman': self._kalman_filter,
            'lowpass_butterworth': self._butterworth_lowpass_filter,
            'savitzky_golay': self._savitzky_golay_filter
        }
        
        if config.filter_type not in filter_methods:
            raise ValueError(f"不支持的滤波类型: {config.filter_type}。"
                           f"可用类型: {list(filter_methods.keys())}")
        
        print(f"应用 {config.filter_type} 滤波器...")
        filtered_data = filter_methods[config.filter_type](data, config)
        
        # 记录滤波历史
        history_key = f"{config.filter_type}_{hash(config)}"
        self.filter_history[history_key] = {
            'config': config,
            'input_length': len(data),
            'output_length': len(filtered_data),
            'reduction_ratio': 1.0 - (np.sum(np.abs(data[:len(filtered_data)] - filtered_data)) /
                                      (np.sum(np.abs(data[:len(filtered_data)])) + 1e-10))
        }
        
        return filtered_data

    def _moving_average_filter(self, 
                              data: np.ndarray,
                              config: FilterConfig) -> np.ndarray:
        """滑动平均滤波器
        
        对数据应用简单的滑动窗口平均。每个输出点是窗口内所有输入点的算术平均值。
        
        数学公式：
            y[i] = (1/N) * sum(x[i-k]) for k in 0..N-1
            
        其中N为窗口大小。
        
        特点：
            - 简单高效，计算复杂度O(n)
            - 会引入相位延迟（约窗口大小/2个采样点）
            - 有效抑制高频随机噪声
            - 可能导致信号边缘效应
        
        Args:
            data (np.ndarray): 输入数据
            config (FilterConfig): 配置（主要使用window_size参数）
            
        Returns:
            np.ndarray: 滤波后数据
        """
        window_size = max(1, config.window_size)
        # 使用scipy的uniform_filter1d，处理边缘模式为'reflect'
        filtered = uniform_filter1d(data.astype(float), size=window_size, mode='reflect')
        return filtered

    def _weighted_moving_average_filter(self,
                                       data: np.ndarray,
                                       config: FilterConfig) -> np.ndarray:
        """加权滑动平均滤波器
        
        与普通滑动平均类似，但窗口内的点具有不同的权重。
        通常越靠近中心的点权重越大。
        
        权重计算：使用线性递减或三角权重
            w[k] = (N - |k - center|) for k in 0..N-1
            
        优点：
            - 减少相位延迟
            - 更好地保留信号的局部特征
        
        Args:
            data (np.ndarray): 输入数据
            config (FilterConfig): 配置
            
        Returns:
            np.ndarray: 滤波后数据
        """
        window_size = max(1, config.window_size)
        n = len(data)
        filtered = np.zeros(n)
        half_window = window_size // 2
        
        # 创建三角权重
        weights = np.array([min(i+1, window_size-i) for i in range(window_size)], dtype=float)
        weights = weights / weights.sum()
        
        # 应用卷积
        padded_data = np.pad(data, half_window, mode='reflect')
        for i in range(n):
            filtered[i] = np.sum(padded_data[i:i+window_size] * weights)
        
        return filtered

    def _exponential_moving_average_filter(self,
                                          data: np.ndarray,
                                          config: FilterConfig) -> np.ndarray:
        """指数加权移动平均滤波器（EMA/指数平滑）
        
        给予最近的数据点更高的权重，权重呈指数衰减。
        
        递推公式：
            y[i] = alpha * x[i] + (1-alpha) * y[i-1]
            
        其中alpha为平滑因子（0 < alpha <= 1）：
            - alpha接近1：响应快，平滑效果弱
            - alpha接近0：响应慢，平滑效果好
        
        特点：
            - 无需指定窗口大小
            - 计算效率高
            - 适合实时处理
            - 对突变响应较快
        
        Args:
            data (np.ndarray): 输入数据
            config (FilterConfig): 配置（使用alpha参数，默认0.3）
            
        Returns:
            np.ndarray: 滤波后数据
        """
        alpha = np.clip(config.alpha, 0.001, 1.0)
        n = len(data)
        filtered = np.zeros(n)
        filtered[0] = data[0]
        
        for i in range(1, n):
            filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i-1]
        
        return filtered

    def _median_filter(self,
                      data: np.ndarray,
                      config: FilterConfig) -> np.ndarray:
        """中值滤波器
        
        用窗口内数据的中位数替代中心点的值。
        对于脉冲噪声（椒盐噪声）特别有效。
        
        数学定义：
            y[i] = median(x[i-k]) for k in -(N-1)/2 .. (N-1)/2
            
        特点：
            - 有效去除脉冲噪声
            - 保持边缘锐利（不像平均那样模糊边缘）
            - 计算量较大（需要排序操作）
            - 对高斯噪声效果不如平均滤波
        
        Args:
            data (np.ndarray): 输入数据
            config (FilterConfig): 配置（使用window_size参数）
            
        Returns:
            np.ndarray: 滤波后数据
        """
        from scipy.ndimage import median_filter
        window_size = max(1, config.window_size)
        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1
        filtered = median_filter(data.astype(float), size=window_size, mode='reflect')
        return filtered

    def _kalman_filter(self,
                      data: np.ndarray,
                      config: FilterConfig) -> np.ndarray:
        """卡尔曼滤波器
        
        使用一维卡尔曼滤波器进行最优状态估计。
        在最小均方误差意义上最优（假设高斯噪声）。
        
        详细原理参见KalmanFilter1D类的文档。
        
        参数调优建议：
            - 如果测量噪声大（传感器不准）：增大R
            - 如果系统变化快：增大Q
            - 如果初始估计不确定：增大P初始值
        
        Args:
            data (np.ndarray): 输入数据
            config (FilterConfig): 配置（使用process_noise, measurement_noise等）
            
        Returns:
            np.ndarray: 滤波后数据
        """
        kf = KalmanFilter1D(
            process_noise=config.process_noise,
            measurement_noise=config.measurement_noise,
            estimated_error=config.estimated_error
        )
        return kf.filter(data)

    def _butterworth_lowpass_filter(self,
                                   data: np.ndarray,
                                   config: FilterConfig) -> np.ndarray:
        """巴特沃斯低通滤波器
        
        IIR滤波器，在通带内具有最大平坦的幅频特性。
        能够有效滤除高于截止频率的成分。
        
        设计参数：
            - cutoff_freq: 截止频率（Hz），在此频率处衰减-3dB
            - order: 滤波器阶数，阶数越高过渡带越陡峭但相位延迟越大
            
        适用场景：
            - 去除高频噪声同时保留低频趋势
            - 传感器信号预处理
            - 需要特定频率响应的应用
        
        注意事项：
            - 假设采样率为100Hz（可根据实际情况调整）
            - 高阶滤波可能导致振铃效应
            - 使用filtfilt实现零相位延迟
        
        Args:
            data (np.ndarray): 输入数据
            config (FilterConfig): 配置（使用cutoff_freq, order参数）
            
        Returns:
            np.ndarray: 滤波后数据
        """
        # 假设采样率（实际应用中应根据时间戳计算）
        sample_rate = 100.0  # Hz，可配置
        
        nyquist_freq = 0.5 * sample_rate
        normalized_cutoff = config.cutoff_freq / nyquist_freq
        
        # 确保归一化截止频率在有效范围内 (0, 1)
        normalized_cutoff = np.clip(normalized_cutoff, 0.001, 0.999)
        
        # 设计巴特沃斯滤波器
        b, a = signal.butter(config.order, normalized_cutoff, btype='low')
        
        # 使用filtfilt实现零相位滤波（前向+反向）
        filtered = signal.filtfilt(b, a, data)
        
        return filtered

    def _savitzky_golay_filter(self,
                              data: np.ndarray,
                              config: FilterConfig) -> np.ndarray:
        """Savitzky-Golay滤波器
        
        基于局部多项式拟合的平滑滤波器。
        在保持信号形状特征的同时去除噪声。
        
        优点：
            - 很好地保留峰值高度和宽度
            - 保持高阶矩不变
            - 适用于光谱数据等
        
        工作原理：
            在每个窗口内拟合一个多项式，用多项式在中心点的值作为输出。
        
        Args:
            data (np.ndarray): 输入数据
            config (FilterConfig): 配置（使用window_size参数，多项式阶数为window_size//4）
            
        Returns:
            np.ndarray: 滤波后数据
        """
        from scipy.signal import savgol_filter
        
        window_size = max(3, config.window_size)
        # 确保窗口大小为奇数且大于多项式阶数
        if window_size % 2 == 0:
            window_size += 1
        
        # 多项式阶数（通常取窗口大小的1/4左右，但不超过窗口-1）
        poly_order = min(window_size // 4, window_size - 2)
        poly_order = max(poly_order, 2)  # 至少2阶
        
        try:
            filtered = savgol_filter(data, window_length=window_size, 
                                    polyorder=poly_order, mode='reflect')
        except Exception as e:
            print(f"Savitzky-Golay滤波失败: {e}，回退到滑动平均")
            filtered = self._moving_average_filter(data, config)
        
        return filtered

    def apply_filter_to_pose_data(self,
                                 pose_data: Any,
                                 config: FilterConfig,
                                 fields: Optional[List[str]] = None) -> Any:
        """对PoseData对象的多个字段同时应用滤波
        
        批量处理位姿数据的各个分量（x,y,z,roll,pitch,yaw），
        返回新的PoseData对象。
        
        Args:
            pose_data: PoseData对象
            config (FilterConfig): 滤波配置
            fields (Optional[List[str]]): 要滤波的字段列表，None表示全部字段
            
        Returns:
            新的PoseData对象，包含滤波后的数据
            
        字段列表：
            ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
            
        示例：
            >>> config = FilterConfig('moving_average', window_size=7)
            >>> filtered_pose = processor.apply_filter_to_pose_data(pose_data, config)
        """
        from bag_parser import PoseData
        import copy
        
        # 复制原始数据结构
        filtered_pose = PoseData(
            timestamp=pose_data.timestamp.copy() if hasattr(pose_data, 'timestamp') else np.array([]),
            x=pose_data.x.copy() if hasattr(pose_data, 'x') else np.array([]),
            y=pose_data.y.copy() if hasattr(pose_data, 'y') else np.array([]),
            z=pose_data.z.copy() if hasattr(pose_data, 'z') else np.array([]),
            roll=pose_data.roll.copy() if hasattr(pose_data, 'roll') else np.array([]),
            pitch=pose_data.pitch.copy() if hasattr(pose_data, 'pitch') else np.array([]),
            yaw=pose_data.yaw.copy() if hasattr(pose_data, 'yaw') else np.array([]),
            quaternion_w=pose_data.quaternion_w.copy() if hasattr(pose_data, 'quaternion_w') else np.array([]),
            quaternion_x=pose_data.quaternion_x.copy() if hasattr(pose_data, 'quaternion_x') else np.array([]),
            quaternion_y=pose_data.quaternion_y.copy() if hasattr(pose_data, 'quaternion_y') else np.array([]),
            quaternion_z=pose_data.quaternion_z.copy() if hasattr(pose_data, 'quaternion_z') else np.array([])
        )
        
        # 确定要处理的字段
        all_fields = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        fields_to_process = fields if fields else [f for f in all_fields 
                                                    if hasattr(pose_data, f) and 
                                                    len(getattr(pose_data, f)) > 0]
        
        # 对每个字段应用滤波
        for field_name in fields_to_process:
            field_data = getattr(pose_data, field_name)
            if len(field_data) > 0:
                filtered_field = self.apply_filter(field_data, config)
                setattr(filtered_pose, field_name, filtered_field)
                print(f"  已滤波字段: {field_name}")
        
        return filtered_pose

    @staticmethod
    def modify_data_point(data: np.ndarray,
                         index: int,
                         new_value: float) -> np.ndarray:
        """修改数据中的单个点
        
        Args:
            data (np.ndarray): 原始数据数组
            index (int): 要修改的点索引
            new_value (float): 新的值
            
        Returns:
            np.ndarray: 修改后的数据副本
            
        Raises:
            IndexError: 当索引超出范围时抛出
        """
        if index < 0 or index >= len(data):
            raise IndexError(f"索引 {index} 超出范围 [0, {len(data)-1}]")
        
        modified = data.copy()
        modified[index] = new_value
        return modified

    @staticmethod
    def remove_outliers(data: np.ndarray,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """检测并剔除异常值
        
        支持多种异常值检测方法。
        
        Args:
            data (np.ndarray): 输入数据
            method (str): 检测方法 ('iqr', 'zscore', 'modified_zscore')
            threshold (float): 异常值判定阈值
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (清洗后的数据, 异常值掩码)
            
        方法说明：
            - IQR: 四分位距方法，基于Q1和Q3
            - Z-Score: 标准分数方法，假设正态分布
            - Modified Z-Score: 基于MAD的鲁棒方法
        """
        mask = np.ones(len(data), dtype=bool)
        
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (data >= lower_bound) & (data <= upper_bound)
            
        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                z_scores = np.abs((data - mean) / std)
                mask = z_scores <= threshold
                
        elif method == 'modified_zscore':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            if mad > 0:
                modified_z = 0.6745 * (data - median) / mad
                mask = np.abs(modified_z) <= threshold
        
        cleaned_data = data[mask].copy()
        outlier_count = np.sum(~mask)
        print(f"已检测并移除 {outlier_count} 个异常值 ({method}方法)")
        
        return cleaned_data, mask

    @staticmethod
    def interpolate_missing(data: np.ndarray,
                           missing_mask: np.ndarray,
                           method: str = 'linear') -> np.ndarray:
        """插值填补缺失的数据点
        
        Args:
            data (np.ndarray): 包含缺失值的数据
            missing_mask (np.ndarray): 缺失值标记（True表示有效，False表示缺失）
            method (str): 插值方法 ('linear', 'cubic', 'nearest')
            
        Returns:
            np.ndarray: 插补完成的数据
        """
        from scipy.interpolate import interp1d
        
        result = data.copy()
        valid_indices = np.where(missing_mask)[0]
        invalid_indices = np.where(~missing_mask)[0]
        
        if len(valid_indices) < 2 or len(invalid_indices) == 0:
            return result
        
        valid_values = data[valid_indices]
        
        try:
            if method == 'cubic' and len(valid_indices) >= 4:
                f = interp1d(valid_indices, valid_values, kind='cubic',
                            bounds_error=False, fill_value='extrapolate')
            elif method == 'nearest':
                f = interp1d(valid_indices, valid_values, kind='nearest',
                            bounds_error=False, fill_value='extrapolate')
            else:  # linear (default)
                f = interp1d(valid_indices, valid_values, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
            
            result[invalid_indices] = f(invalid_indices)
            print(f"已使用{method}插值法填补 {len(invalid_indices)} 个缺失点")
            
        except Exception as e:
            print(f"插值失败: {e}，保持原数据")
        
        return result

    def get_filter_comparison_stats(self,
                                   original: np.ndarray,
                                   filtered: np.ndarray) -> Dict[str, float]:
        """计算滤波前后的统计对比指标
        
        Args:
            original (np.ndarray): 原始数据
            filtered (np.ndarray): 滤波后数据
            
        Returns:
            Dict[str, float]: 包含各种统计指标的字典
            
        统计指标包括：
            - mean_reduction: 均值变化百分比
            - std_reduction: 标准差降低百分比（衡量平滑程度）
            - noise_reduction: 噪声降低程度估计
            - signal_preservation: 信号保留程度
            - rmse: 原始与滤波间的均方根误差
        """
        # 确保长度一致
        min_len = min(len(original), len(filtered))
        orig = original[:min_len]
        filt = filtered[:min_len]
        
        stats = {}
        
        # 基本统计
        stats['original_mean'] = np.mean(orig)
        stats['filtered_mean'] = np.mean(filt)
        stats['original_std'] = np.std(orig)
        stats['filtered_std'] = np.std(filt)
        
        # 平滑度改善（标准差降低比例）
        if stats['original_std'] > 0:
            stats['std_reduction_pct'] = (stats['original_std'] - stats['filtered_std']) / stats['original_std'] * 100
        else:
            stats['std_reduction_pct'] = 0.0
        
        # RMSE（衡量滤波引入的变化）
        stats['rmse'] = np.sqrt(np.mean((orig - filt) ** 2))
        
        # 信噪比改善估算（简化版本）
        if stats['rmse'] > 0 and stats['original_std'] > 0:
            # 假设原始信号中的高频成分主要是噪声
            stats['estimated_snr_improvement_db'] = 10 * np.log10(
                stats['original_std']**2 / (stats['filtered_std']**2 + 1e-10)
            )
        else:
            stats['estimated_snr_improvement_db'] = 0.0
        
        # 相关性（衡量信号形状保留程度）
        if np.std(orig) > 0 and np.std(filt) > 0:
            correlation = np.corrcoef(orig, filt)[0, 1]
            stats['correlation'] = correlation
        else:
            stats['correlation'] = 1.0
        
        return stats

    def print_filter_report(self,
                          original: np.ndarray,
                          filtered: np.ndarray,
                          filter_name: str = "") -> None:
        """打印详细的滤波效果报告
        
        Args:
            original (np.ndarray): 原始数据
            filtered (np.ndarray): 滤波后数据
            filter_name (str): 滤波器名称（用于报告标题）
        """
        stats = self.get_filter_comparison_stats(original, filtered)
        
        print("\n" + "=" * 60)
        print(f"滤波效果报告 - {filter_name}" if filter_name else "滤波效果报告")
        print("=" * 60)
        print(f"\n基本统计:")
        print(f"  原始数据均值:     {stats.get('original_mean', 0):.6f}")
        print(f"  滤波后均值:       {stats.get('filtered_mean', 0):.6f}")
        print(f"  原始数据标准差:   {stats.get('original_std', 0):.6f}")
        print(f"  滤波后标准差:     {stats.get('filtered_std', 0):.6f}")
        print(f"\n滤波效果:")
        print(f"  平滑度提升:       {stats.get('std_reduction_pct', 0):.2f}%")
        print(f"  均方根误差(RMSE): {stats.get('rmse', 0):.6f}")
        print(f"  信号相关性:       {stats.get('correlation', 0):.4f}")
        if stats.get('estimated_snr_improvement_db', 0) != 0:
            print(f"  估算SNR改善:     {stats.get('estimated_snr_improvement_db', 0):.2f} dB")
        print("=" * 60)


def main():
    """测试函数 - FilterProcessor模块的功能演示"""
    print("=" * 60)
    print("滤波处理器测试")
    print("=" * 60)
    
    # 生成包含噪声的测试数据
    np.random.seed(42)
    n_points = 200
    t = np.linspace(0, 4*np.pi, n_points)
    
    # 原始信号：正弦波 + 趋势
    clean_signal = np.sin(t) + 0.1*t
    # 添加噪声
    noisy_signal = clean_signal + np.random.normal(0, 0.3, n_points)
    # 添加一些异常值
    noisy_signal[50] += 3.0
    noisy_signal[150] -= 2.5
    
    print(f"生成测试数据: {n_points} 个点")
    print(f"原始标准差: {np.std(clean_signal):.4f}")
    print(f"含噪标准差: {np.std(noisy_signal):.4f}")
    
    processor = FilterProcessor()
    
    # 测试不同的滤波器
    filters_to_test = [
        ('滑动平均', FilterConfig(filter_type='moving_average', window_size=7)),
        ('加权滑动平均', FilterConfig(filter_type='weighted_moving_average', window_size=7)),
        ('指数平滑', FilterConfig(filter_type='exponential_moving_average', alpha=0.3)),
        ('中值滤波', FilterConfig(filter_type='median', window_size=5)),
        ('卡尔曼滤波', FilterConfig(filter_type='kalman', process_noise=0.01, measurement_noise=0.1)),
        ('巴特沃斯低通', FilterConfig(filter_type='lowpass_butterworth', cutoff_freq=2.0)),
        ('Savitzky-Golay', FilterConfig(filter_type='savitzky_golay', window_size=11))
    ]
    
    results = {}
    
    print("\n开始测试各滤波器...\n")
    
    for name, config in filters_to_test:
        try:
            filtered = processor.apply_filter(noisy_signal, config)
            results[name] = filtered
            
            # 打印简要统计
            std_red = (np.std(noisy_signal) - np.std(filtered)) / np.std(noisy_signal) * 100
            print(f"✓ {name:15s}: 标准差降低 {std_red:6.2f}%")
            
            # 打印详细报告（仅对前3种）
            if list(filters_to_test).index((name, config)) < 3:
                processor.print_filter_report(noisy_signal, filtered, name)
                
        except Exception as e:
            print(f"✗ {name:15s}: 错误 - {str(e)}")
    
    # 测试异常值剔除
    print("\n测试异常值剔除功能...")
    cleaned, mask = processor.remove_outliers(noisy_signal, method='iqr')
    print(f"  移除异常值: {np.sum(~mask)} 个")
    print(f"  清洗后数据长度: {len(cleaned)}")
    
    # 测试数据修改
    print("\n测试数据点修改功能...")
    modified = processor.modify_data_point(noisy_signal, 50, clean_signal[50])
    print(f"  修改第50个点: {noisy_signal[50]:.4f} -> {modified[50]:.4f}")
    
    print("\n所有测试完成！")


if __name__ == "__main__":
    main()
