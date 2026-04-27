# -*- coding: utf-8 -*-
"""
ROS Bag文件数据提取模块
功能：解析.bag文件，提取话题信息和消息数据
作者：Auto-generated
日期：2026-04-27

支持的后端：
1. rosbag（推荐）- 传统ROS包，需要完整ROS环境，兼容性最佳
2. rosbags - 纯Python实现，无需ROS环境（备用方案）

使用方法：
    # 有ROS环境（推荐）
    from bag_parser import BagParser
    parser = BagParser('example.bag', backend='rosbag')
    
    # 无ROS环境（备用）
    from bag_parser import BagParser
    parser = BagParser('example.bag', backend='rosbags')
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


def _try_import_rosbag():
    """尝试导入rosbag库（需要ROS环境）"""
    try:
        import rosbag
        return True
    except ImportError:
        return False


def _try_import_rosbags():
    """尝试导入rosbags库（纯Python实现）"""
    try:
        from rosbags.rosbag1.reader import Reader
        return True
    except ImportError:
        return False


# 检测可用的后端（优先检测rosbag）
AVAILABLE_BACKENDS = []
if _try_import_rosbag():
    AVAILABLE_BACKENDS.append('rosbag')
if _try_import_rosbags():
    AVAILABLE_BACKENDS.append('rosbags')


@dataclass
class TopicInfo:
    """话题信息数据类
    
    存储单个话题的基本信息，包括名称、类型、消息数量和时间范围
    
    Attributes:
        name (str): 话题名称
        msg_type (str): 消息类型（如nav_msgs/Odometry）
        message_count (int): 该话题的消息总数
        start_time (float): 首条消息的时间戳（秒）
        end_time (float): 末条消息的时间戳（秒）
        duration (float): 数据持续时间（秒）
    """
    name: str
    msg_type: str
    message_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    def __post_init__(self):
        """初始化后处理：计算持续时间"""
        if self.end_time > self.start_time:
            self.duration = self.end_time - self.start_time


@dataclass
class PoseData:
    """位姿数据类
    
    存储从ROS消息中提取的位置和姿态数据
    
    Attributes:
        timestamp (np.ndarray): 时间戳数组
        x (np.ndarray): X坐标数组
        y (np.ndarray): Y坐标数组
        z (np.ndarray): Z坐标数组
        roll (np.ndarray): 翻滚角数组（弧度）
        pitch (np.ndarray): 俯仰角数组（弧度）
        yaw (np.ndarray): 偏航角数组（弧度）
        quaternion_w (np.ndarray): 四元数w分量
        quaternion_x (np.ndarray): 四元数x分量
        quaternion_y (np.ndarray): 四元数y分量
        quaternion_z (np.ndarray): 四元数z分量
    """
    timestamp: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    z: np.ndarray = field(default_factory=lambda: np.array([]))
    roll: np.ndarray = field(default_factory=lambda: np.array([]))
    pitch: np.ndarray = field(default_factory=lambda: np.array([]))
    yaw: np.ndarray = field(default_factory=lambda: np.array([]))
    quaternion_w: np.ndarray = field(default_factory=lambda: np.array([]))
    quaternion_x: np.ndarray = field(default_factory=lambda: np.array([]))
    quaternion_y: np.ndarray = field(default_factory=lambda: np.array([]))
    quaternion_z: np.ndarray = field(default_factory=lambda: np.array([]))


class BagParser:
    """Bag文件解析器主类
    
    提供完整的.bag文件解析功能，包括：
    - 文件读取与验证
    - 话题信息提取
    - 特定话题数据解析
    - 数据导出功能
    
    支持两种后端：
    - 'rosbag': 传统ROS包（推荐，需要完整ROS环境，兼容性最佳）
    - 'rosbags': 纯Python实现（备用方案，无需ROS环境）
    
    使用示例：
        >>> parser = BagParser('example.bag')  # 自动选择可用后端（优先rosbag）
        >>> parser.parse_bag()
        >>> topics = parser.get_all_topics()
        >>> pose_data = parser.extract_pose_data('/Odometry')
    
    注意事项：
        - 推荐使用rosbag后端，兼容性和稳定性最好
        - 大文件解析可能需要较长时间
        - 自动选择最合适的后端（优先rosbag）
    """

    def __init__(self, bag_file_path: str, backend: Optional[str] = None):
        """初始化Bag解析器
        
        Args:
            bag_file_path (str): .bag文件的完整路径
            backend (Optional[str]): 指定后端类型 ('rosbag' 或 'rosbags')
                                    默认为None，自动选择可用后端（优先rosbag）
            
        Raises:
            FileNotFoundError: 当文件不存在时抛出
            ValueError: 当文件扩展名不是.bag或没有可用后端时抛出
        """
        self.bag_file_path = bag_file_path
        self.topics_info: Dict[str, TopicInfo] = {}
        self.raw_data: Dict[str, List[Tuple[float, Any]]] = defaultdict(list)
        self.pose_data_cache: Dict[str, PoseData] = {}
        
        # 选择后端（默认优先使用rosbag）
        if backend is None:
            backend = self._select_backend()
        
        if backend not in AVAILABLE_BACKENDS:
            raise ValueError(
                f"指定的后端 '{backend}' 不可用。"
                f"可用后端: {AVAILABLE_BACKENDS}。\n"
                f"推荐安装：sudo apt-get install ros-noetic-rosbag （Ubuntu/ROS Noetic）\n"
                f"或确保已配置完整的ROS环境"
            )
        
        self.backend = backend
        
        self._validate_bag_file()

    def _select_backend(self) -> str:
        """自动选择最合适的后端（优先选择rosbag）"""
        if 'rosbag' in AVAILABLE_BACKENDS:
            print("✅ 使用后端: rosbag (传统ROS包，兼容性最佳)")
            return 'rosbag'
        elif 'rosbags' in AVAILABLE_BACKENDS:
            print("⚠️  使用后端: rosbags (纯Python实现，兼容性有限)")
            return 'rosbags'
        else:
            raise ValueError(
                "未找到可用的bag解析库！\n"
                "请安装以下任一库：\n"
                "  1. 推荐：配置完整ROS环境（包含rosbag）\n"
                "     Ubuntu: sudo apt-get install ros-noetic-rosbag\n"
                "  2. 备用：pip install rosbags (纯Python，无需ROS环境)"
            )

    def _validate_bag_file(self) -> None:
        """验证bag文件的有效性
        
        检查文件是否存在且扩展名正确
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不正确
        """
        if not os.path.exists(self.bag_file_path):
            raise FileNotFoundError(f"Bag文件不存在: {self.bag_file_path}")
        
        if not self.bag_file_path.endswith('.bag'):
            raise ValueError(f"文件格式错误，期望.bag文件: {self.bag_file_path}")

    def parse_bag(self) -> Dict[str, TopicInfo]:
        """解析bag文件并提取所有话题信息
        
        读取整个bag文件，收集每个话题的基本统计信息，
        包括消息类型、数量和时间范围。
        
        Returns:
            Dict[str, TopicInfo]: 以话题名为键的TopicInfo字典
            
        示例：
            >>> topics = parser.parse_bag()
            >>> for name, info in topics.items():
            ...     print(f"{name}: {info.message_count} 条消息")
        
        注意事项：
            - 对于大型bag文件，此操作可能需要几分钟
            - 结果会缓存在self.topics_info中
        """
        print(f"\n正在解析bag文件: {self.bag_file_path}")
        print(f"使用后端: {self.backend}")
        
        try:
            if self.backend == 'rosbag':
                self._parse_with_rosbag()
            else:
                self._parse_with_rosbags()
            
            print(f"\n✅ 解析完成！共发现 {len(self.topics_info)} 个话题\n")
            
        except Exception as e:
            print(f"❌ 解析bag文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        return self.topics_info

    def _parse_with_rosbag(self) -> None:
        """使用传统rosbag库解析bag文件（推荐方案）
        
        这是ROS官方提供的标准接口，具有最佳的兼容性和稳定性。
        需要完整ROS环境支持。
        """
        import rosbag
        
        print("  正在使用传统rosbag库读取文件...")
        
        bag = rosbag.Bag(self.bag_file_path, 'r')
        
        try:
            # 获取基本信息
            info = bag.get_type_and_topic_info()
            
            # 解析每个话题的信息
            for topic_name, topic_info in info.topics.items():
                topic_info_obj = TopicInfo(
                    name=topic_name,
                    msg_type=topic_info.msg_type,
                    message_count=topic_info.message_count
                )
                
                # 收集该话题的所有消息数据和时间戳
                timestamps = []
                message_count = 0
                
                for topic, msg, t in bag.read_messages(topics=[topic_name]):
                    timestamps.append(t.to_sec())
                    self.raw_data[topic_name].append((t.to_sec(), msg))
                    message_count += 1
                
                if timestamps:
                    topic_info_obj.start_time = min(timestamps)
                    topic_info_obj.end_time = max(timestamps)
                    topic_info_obj.__post_init__()
                
                self.topics_info[topic_name] = topic_info_obj
                print(f"  ✓ 已解析话题: {topic_name} ({topic_info.msg_type}) - {message_count} 条消息")
                
        finally:
            bag.close()

    def _parse_with_rosbags(self) -> None:
        """使用rosbags库解析bag文件（备用方案）
        
        rosbags是纯Python实现的rosbag读取器，不需要ROS环境。
        但可能存在某些兼容性问题。
        """
        try:
            from rosbags.rosbag1.reader import Reader
        except ImportError:
            raise ImportError(
                "rosbags库导入失败！请确保已正确安装rosbags库:\n"
                "pip install rosbags"
            )
        
        print("  正在使用rosbags库读取文件...")
        
        with Reader(self.bag_file_path) as reader:
            connections = reader.connections
            
            for connection in connections:
                topic_name = connection.topic
                msg_type = connection.msgtype
                
                timestamps = []
                message_count = 0
                
                for connection_msg, timestamp, rawdata in reader.messages():
                    if connection_msg.topic == topic_name:
                        timestamps.append(timestamp / 1e9)
                        message_count += 1
                        self.raw_data[topic_name].append((timestamp / 1e9, rawdata))
                
                topic_info_obj = TopicInfo(
                    name=topic_name,
                    msg_type=msg_type,
                    message_count=message_count
                )
                
                if timestamps:
                    topic_info_obj.start_time = min(timestamps)
                    topic_info_obj.end_time = max(timestamps)
                    topic_info_obj.__post_init__()
                
                self.topics_info[topic_name] = topic_info_obj
                print(f"  ✓ 已解析话题: {topic_name} ({msg_type}) - {message_count} 条消息")

    def get_all_topics(self) -> List[TopicInfo]:
        """获取所有话题信息的列表形式
        
        Returns:
            List[TopicInfo]: 话题信息对象列表
        """
        return list(self.topics_info.values())

    def get_topic_names(self) -> List[str]:
        """获取所有话题名称列表
        
        Returns:
            List[str]: 话题名称字符串列表
        """
        return list(self.topics_info.keys())

    def get_topic_info(self, topic_name: str) -> Optional[TopicInfo]:
        """获取指定话题的详细信息
        
        Args:
            topic_name (str): 话题名称
            
        Returns:
            Optional[TopicInfo]: 话题信息对象，如果不存在则返回None
        """
        return self.topics_info.get(topic_name)

    def extract_pose_data(self, topic_name: str) -> Optional[PoseData]:
        """从指定话题提取位姿数据
        
        从包含位姿信息的话题（如Odometry、PoseStamped等）中
        提取位置(x,y,z)和姿态(roll,pitch,yaw)数据。
        
        Args:
            topic_name (str): 要提取数据的话题名称
            
        Returns:
            Optional[PoseData]: 位姿数据对象，如果提取失败返回None
            
        支持的消息类型：
            - nav_msgs/Odometry
            - geometry_msgs/PoseStamped
            - nav_msgs/Path（提取路径点）
            
        注意事项：
            - 四元数会自动转换为欧拉角（roll, pitch, yaw）
            - 结果会被缓存以提高重复访问性能
            - rosbag后端直接从内存中的消息对象提取，性能最优
        """
        # 检查缓存
        if topic_name in self.pose_data_cache:
            return self.pose_data_cache[topic_name]
        
        if topic_name not in self.raw_data or len(self.raw_data[topic_name]) == 0:
            print(f"⚠️  警告: 话题 '{topic_name}' 未找到或无数据")
            return None
        
        pose_data = PoseData()
        timestamps = []
        positions_x, positions_y, positions_z = [], [], []
        quaternions_w, quaternions_x, quaternions_y, quaternions_z = [], [], [], []
        
        success_count = 0
        error_count = 0
        
        try:
            messages = self.raw_data[topic_name]
            
            print(f"\n正在从话题 '{topic_name}' 提取位姿数据...")
            print(f"  消息总数: {len(messages)}")
            
            for idx, (timestamp, msg) in enumerate(messages):
                try:
                    timestamps.append(timestamp)
                    
                    # 对于rosbags后端，需要先反序列化消息
                    if self.backend == 'rosbags':
                        msg = self._deserialize_rosbags_message(msg, topic_name)
                        if msg is None:
                            error_count += 1
                            continue
                    
                    # 从消息对象中提取位姿数据
                    pose = self._extract_pose_from_message(msg)
                    
                    if pose is not None:
                        pos, quat = pose
                        positions_x.append(pos[0])
                        positions_y.append(pos[1])
                        positions_z.append(pos[2])
                        quaternions_w.append(quat[0])
                        quaternions_x.append(quat[1])
                        quaternions_y.append(quat[2])
                        quaternions_z.append(quat[3])
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 5 or error_count % 100 == 0:
                        print(f"  ⚠️  处理消息 #{idx+1} 时出错: {str(e)[:80]}")
                    continue
            
            # 打印统计信息
            total = success_count + error_count
            print(f"\n  📊 位姿数据提取统计:")
            print(f"     ✓ 成功: {success_count}/{total}")
            if error_count > 0:
                print(f"     ✗ 失败/跳过: {error_count}/{total}")
                success_rate = (success_count / total * 100) if total > 0 else 0
                print(f"     成功率: {success_rate:.1f}%")
            
            if success_count == 0 and total > 0:
                print("  ❌ 所有消息都无法提取位姿数据，可能是:")
                print("     - 消息类型不包含位姿信息")
                print("     - 消息结构不符合预期")
                print("     - 数据损坏或格式异常")
                return None
            
            # 转换为numpy数组
            pose_data.timestamp = np.array(timestamps[:success_count])
            pose_data.x = np.array(positions_x)
            pose_data.y = np.array(positions_y)
            pose_data.z = np.array(positions_z)
            pose_data.quaternion_w = np.array(quaternions_w)
            pose_data.quaternion_x = np.array(quaternions_x)
            pose_data.quaternion_y = np.array(quaternions_y)
            pose_data.quaternion_z = np.array(quaternions_z)
            
            # 将四元数转换为欧拉角
            if len(pose_data.quaternion_w) > 0:
                pose_data.roll, pose_data.pitch, pose_data.yaw = \
                    self._quaternion_to_euler(
                        pose_data.quaternion_w,
                        pose_data.quaternion_x,
                        pose_data.quaternion_y,
                        pose_data.quaternion_z
                    )
            
            # 缓存结果
            self.pose_data_cache[topic_name] = pose_data
            print(f"\n✅ 成功提取话题 '{topic_name}' 的位姿数据: {success_count} 个有效数据点")
            
            return pose_data
            
        except Exception as e:
            print(f"❌ 提取位姿数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _deserialize_rosbags_message(self, rawdata: Any, topic_name: str) -> Optional[Any]:
        """反序列化rosbags原始消息数据（仅用于rosbags后端）
        
        Args:
            rawdata: 原始二进制数据
            topic_name: 话题名称（用于查找消息类型）
            
        Returns:
            反序列化后的消息对象，失败返回None
        """
        try:
            from rosbags.serde import deserialize_cdr
            
            # 从topics_info获取消息类型
            if topic_name in self.topics_info:
                msg_type = self.topics_info[topic_name].msg_type
                msg = deserialize_cdr(rawdata, msg_type)
                return msg
            else:
                print(f"  警告: 无法确定话题 '{topic_name}' 的消息类型")
                return None
                
        except Exception as e:
            print(f"  反序列化消息失败: {str(e)[:80]}")
            return None

    def _extract_pose_from_message(self, msg: Any) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """从ROS消息中提取位置和四元数数据
        
        支持多种常见的包含位姿信息的消息类型：
        - nav_msgs/Odometry: 标准里程计消息
        - geometry_msgs/PoseStamped: 带时间戳的位姿消息
        - nav_msgs/Path: 路径消息（提取第一个点）
        
        Args:
            msg: ROS消息对象
            
        Returns:
            Optional[Tuple]: 包含(位置元组, 四元数元组)的元组，或None
        """
        try:
            # 尝试不同的消息类型结构
            if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                # nav_msgs/Odometry 类型
                pose = msg.pose.pose
                position = (pose.position.x, pose.position.y, pose.position.z)
                quaternion = (pose.orientation.w, pose.orientation.x,
                             pose.orientation.y, pose.orientation.z)
                return (position, quaternion)
                
            elif hasattr(msg, 'pose'):
                # geometry_msgs/PoseStamped 类型
                pose = msg.pose
                position = (pose.position.x, pose.position.y, pose.position.z)
                quaternion = (pose.orientation.w, pose.orientation.x,
                             pose.orientation.y, pose.orientation.z)
                return (position, quaternion)
                
            elif hasattr(msg, 'poses') and len(msg.poses) > 0:
                # nav_msgs/Path 类型 - 返回第一个路径点
                first_pose = msg.poses[0].pose
                position = (first_pose.position.x, first_pose.position.y, first_pose.position.z)
                quaternion = (first_pose.orientation.w, first_pose.orientation.x,
                             first_pose.orientation.y, first_pose.orientation.z)
                return (position, quaternion)
                
            else:
                # 尝试其他可能的字段名组合
                if hasattr(msg, 'position') and hasattr(msg, 'orientation'):
                    position = (msg.position.x, msg.position.y, msg.position.z)
                    quaternion = (msg.orientation.w, msg.orientation.x,
                                 msg.orientation.y, msg.orientation.z)
                    return (position, quaternion)
                else:
                    return None
                
        except Exception:
            return None

    @staticmethod
    def _quaternion_to_euler(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将四元数转换为欧拉角（roll, pitch, yaw）
        
        使用ZYX欧拉角约定（常用于航空领域）：
        - roll: 绕X轴旋转
        - pitch: 绕Y轴旋转  
        - yaw: 绕Z轴旋转
        
        Args:
            w (np.ndarray): 四元数的w分量数组
            x (np.ndarray): 四元数的x分量数组
            y (np.ndarray): 四元数的y分量数组
            z (np.ndarray): 四元数的z分量数组
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (roll数组, pitch数组, yaw数组)，单位为弧度
                
        数学公式：
            roll (x-axis rotation) = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
            pitch (y-axis rotation) = asin(2*(w*y - z*x))
            yaw (z-axis rotation) = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # 处理asin的定义域问题
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return (roll, pitch, yaw)

    def export_topics_to_csv(self, output_dir: str = './exported_data') -> bool:
        """将所有话题信息导出为CSV文件
        
        Args:
            output_dir (str): 输出目录路径，默认为'./exported_data'
            
        Returns:
            bool: 导出是否成功
            
        导出内容：
            - topics_summary.csv: 所有话题的汇总信息
            - 每个话题单独的CSV文件，包含完整的数据记录
        """
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 导出话题汇总信息
            summary_data = []
            for topic_name, info in self.topics_info.items():
                summary_data.append({
                    '话题名称': info.name,
                    '消息类型': info.msg_type,
                    '消息数量': info.message_count,
                    '开始时间': info.start_time,
                    '结束时间': info.end_time,
                    '持续时间': info.duration
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = os.path.join(output_dir, 'topics_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            print(f"已导出话题汇总: {summary_csv_path}")
            
            # 导出每个话题的详细数据
            for topic_name, messages in self.raw_data.items():
                safe_name = topic_name.replace('/', '_').strip('_')
                csv_path = os.path.join(output_dir, f'{safe_name}_data.csv')
                
                data_rows = []
                for timestamp, msg in messages:
                    row = {'timestamp': timestamp}
                    
                    if msg is not None:
                        try:
                            if hasattr(msg, '__slots__'):
                                for slot in msg.__slots__:
                                    value = getattr(msg, slot, None)
                                    row[slot] = self._convert_value_to_str(value)
                            elif hasattr(msg, '_fields'):
                                for field_name in msg._fields:
                                    value = getattr(msg, field_name, None)
                                    row[field_name] = self._convert_value_to_str(value)
                        except Exception:
                            pass
                    
                    data_rows.append(row)
                
                if data_rows:
                    df = pd.DataFrame(data_rows)
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    print(f"  已导出话题数据: {csv_path}")
            
            print(f"\n数据导出完成！输出目录: {output_dir}")
            return True
            
        except Exception as e:
            print(f"导出数据时出错: {str(e)}")
            return False

    @staticmethod
    def _convert_value_to_str(value: Any) -> str:
        """将值转换为字符串表示
        
        用于CSV导出时的值转换，处理特殊类型
        
        Args:
            value: 任意类型的值
            
        Returns:
            str: 字符串形式的值
        """
        if value is None:
            return ''
        elif isinstance(value, (list, tuple)):
            return ';'.join([str(v) for v in value])
        else:
            return str(value)

    def get_statistics_report(self) -> str:
        """生成bag文件的统计报告
        
        Returns:
            str: 格式化的统计报告文本
        """
        report = []
        report.append("=" * 80)
        report.append("ROS Bag文件分析报告")
        report.append("=" * 80)
        report.append(f"文件路径: {self.bag_file_path}")
        report.append(f"使用后端: {self.backend}")
        report.append(f"话题总数: {len(self.topics_info)}")
        report.append("")
        
        total_messages = 0
        for topic_name, info in sorted(self.topics_info.items()):
            report.append(f"话题: {topic_name}")
            report.append(f"  类型: {info.msg_type}")
            report.append(f"  消息数: {info.message_count}")
            report.append(f"  时间范围: {info.start_time:.2f}s - {info.end_time:.2f}s")
            report.append(f"  持续时间: {info.duration:.2f}s")
            report.append("")
            total_messages += info.message_count
        
        report.append("-" * 80)
        report.append(f"总消息数: {total_messages}")
        report.append("=" * 80)
        
        return '\n'.join(report)


def main():
    """测试函数 - BagParser模块的功能演示"""
    print("=" * 60)
    print("ROS Bag文件解析器测试")
    print("=" * 60)
    
    # 显示可用后端
    print(f"\n检测到的可用后端: {AVAILABLE_BACKENDS if AVAILABLE_BACKENDS else '无'}")
    
    example_bag = "example.bag"
    
    if os.path.exists(example_bag):
        try:
            parser = BagParser(example_bag)
            parser.parse_bag()
            
            print("\n发现的话题:")
            for topic in parser.get_all_topics():
                print(f"  - {topic.name} ({topic.msg_type}): {topic.message_count} 条消息")
            
            print("\n" + parser.get_statistics_report())
            
            target_topics = ['/Odometry', '/path', '/robot1/robot/cmd_vel']
            for topic_name in target_topics:
                if topic_name in parser.get_topic_names():
                    pose_data = parser.extract_pose_data(topic_name)
                    if pose_data and len(pose_data.timestamp) > 0:
                        print(f"\n话题 '{topic_name}' 数据预览:")
                        print(f"  时间范围: {pose_data.timestamp[0]:.2f}s - {pose_data.timestamp[-1]:.2f}s")
                        print(f"  X范围: [{pose_data.x.min():.3f}, {pose_data.x.max():.3f}]")
                        print(f"  Y范围: [{pose_data.y.min():.3f}, {pose_data.y.max():.3f}]")
            
        except Exception as e:
            print(f"错误: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"未找到示例bag文件: {example_bag}")
        print("请提供有效的.bag文件路径进行测试")


if __name__ == "__main__":
    main()
