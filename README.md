# ROS Bag文件数据处理与可视化系统

## 📋 项目简介

本项目是一个功能完整的ROS (Robot Operating System) `.bag` 文件数据分析与可视化工具，基于Python开发，提供图形用户界面(GUI)和命令行两种使用方式。

### 主要功能特性

✅ **数据提取模块**
- 完整解析`.bag`文件，自动识别所有ROS话题
- 提取话题基本信息（名称、类型、消息数量、时间戳范围）
- 支持多种消息类型的数据提取：
  - `nav_msgs/Odometry` - 里程计位姿数据
  - `geometry_msgs/PoseStamped` - 位姿消息
  - `geometry_msgs/PoseWithCovarianceStamped` - 带协方差的位姿
  - `nav_msgs/Path` - 路径消息（提取最新位置点）
  - `geometry_msgs/Twist` - 线速度/角速度数据
  - `geometry_msgs/TwistStamped` - 带时间戳的速度数据
  - `geometry_msgs/Vector3Stamped` - 三维向量数据
  - `sensor_msgs/Imu` - IMU数据
- 不支持的话题类型点击时弹窗提示
- 数据导出为CSV格式

✅ **数据可视化模块**
- 位置数据(x, y, z)随时间变化的曲线图
- 姿态数据(roll, pitch, yaw)随时间变化的曲线图
- 2D/3D运动轨迹可视化（3D支持交互旋转/缩放/平移）
- 多话题数据对比显示
- 综合仪表板视图
- 图表内嵌预览与导航工具栏
- 图表包含完整的标题、标签、图例和网格线
- 中文字体自动适配（Windows/Linux/macOS）

✅ **数据修改与滤波模块**
- 7种专业滤波算法：
  - 滑动平均滤波 (Moving Average)
  - 加权滑动平均滤波 (Weighted Moving Average)
  - 指数加权移动平均 (Exponential MA / EMA)
  - 中值滤波 (Median Filter)
  - 卡尔曼滤波 (Kalman Filter)
  - 巴特沃斯低通滤波 (Butterworth Low-pass)
  - Savitzky-Golay滤波
- 可调整的滤波参数（窗口大小、系数、截止频率等）
- 滤波前后数据对比与统计报告
- 异常值检测与剔除功能
- 数据点手动修改接口

✅ **友好的用户界面**
- 基于PyQt5的现代GUI界面
- 直观的操作流程设计
- 实时状态反馈与进度显示
- 多标签页信息展示
- 支持中文界面

---

## 📦 环境配置要求

### 系统要求
- **操作系统**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **Python版本**: Python 3.7 或更高版本（推荐3.8+）
- **内存**: 建议4GB以上（处理大型bag文件时）
- **磁盘空间**: 至少500MB可用空间

### 依赖库安装

#### ⚠️ 重要提示：推荐使用传统rosbag后端（需要完整ROS环境）

本程序**强烈建议**在配置了完整ROS环境的系统中运行，使用传统`rosbag`工具进行数据解析，这是ROS官方提供的标准接口，具有**最佳的兼容性和稳定性**。

如果您的系统没有配置ROS环境，程序会自动降级使用纯Python的`rosbags`库作为备选方案，但可能存在某些兼容性问题。

#### 1. 安装Python环境

确保已安装Python 3.7+。可通过以下命令检查：

```bash
python --version
# 或
python3 --version
```

如果未安装，请从 [Python官网](https://www.python.org/downloads/) 下载安装。

#### 2. 配置ROS环境（推荐方案）

##### 方案A：Ubuntu + ROS Noetic（最推荐）

```bash
# 更新软件包列表
sudo apt-get update

# 安装ROS Noetic的rosbag包
sudo apt-get install ros-noetic-rosbag

# 配置ROS环境变量（每次打开终端都需要执行）
source /opt/ros/noetic/setup.bash

# 验证安装成功
python -c "import rosbag; print('✅ rosbag导入成功')"
```

##### 方案B：Ubuntu + ROS Melodic

```bash
sudo apt-get update
sudo apt-get install ros-melodic-rosbag
source /opt/ros/melodic/setup.bash

# 验证安装
python -c "import rosbag; print('✅ rosbag导入成功')"
```

##### 方案C：Docker容器运行ROS（跨平台通用）

```bash
# 拉取ROS Noetic桌面完整版镜像
docker pull osrf/ros:noetic-desktop-full

# 运行容器并挂载当前目录
docker run -it --rm -v $(pwd):/workspace osrf/ros:noetic-desktop-full

# 在容器内进入工作目录并运行程序
cd /workspace && python main.py
```

##### 方案D：Windows WSL2 + ROS

```bash
# 1. 在Windows上安装WSL2和Ubuntu
# 2. 在WSL2中按照方案A配置ROS Noetic
# 3. 从Windows资源管理器访问Linux文件: \\wsl$\Ubuntu-20.04\home\<user>
```

#### 3. 安装Python依赖包

项目提供了`requirements.txt`文件，包含所有必需的依赖库：

```bash
# 进入项目目录
cd Demo1

# 确保已source ROS环境
source /opt/ros/noetic/setup.bash  # 或对应版本的setup.bash

# 安装所有依赖（推荐使用虚拟环境）
pip install -r requirements.txt
```

**或逐个安装核心依赖**：

```bash
# 数值计算与可视化
pip install numpy matplotlib scipy pandas

# GUI界面
pip install PyQt5

# 其他工具
pip install pyyaml

# 注意：rosbag和rospy已在ROS环境中预装，无需额外pip安装
```

#### 4. 备选方案：无ROS环境（不推荐）

⚠️ **警告**：此方案可能存在兼容性问题，仅当无法使用完整ROS环境时才考虑！

```bash
# 安装纯Python实现的rosbags库
pip install rosbags

# 安装其他基础依赖
pip install numpy matplotlib scipy pandas PyQt5 pyyaml

# 运行时程序会自动检测并使用rosbags后端
python main.py
```

**已知限制**：
- 某些特殊消息类型可能无法正确解析
- 可能遇到UTF-8解码错误
- 解析成功率可能低于100%
- 不支持所有ROS1 bag文件格式

---

## 🚀 快速开始

### 方式一：使用GUI界面（推荐）

#### 启动程序

```bash
python main.py
```

或指定bag文件路径直接启动：

```bash
python main.py -f your_data.bag
```

#### 操作步骤

1. **打开Bag文件**
   - 点击左侧面板"📂 打开Bag文件"按钮
   - 或使用菜单 `文件 → 打开Bag文件...` (快捷键 Ctrl+O)
   - 选择要分析的`.bag`文件

2. **浏览话题列表**
   - 加载完成后，左侧显示所有发现的话题
   - 每个话题显示名称、消息类型和消息数量
   - 右侧"📝 信息"标签页显示详细统计报告

3. **选择目标话题进行数据分析**
   
   **方法A - 使用快速按钮**：
   - 点击左侧"🎯 常用目标话题"区域中的预设话题按钮
   - 支持: `/Odometry`, `/path`, `/path_exp`, `/robot1/robot/cmd_vel`, `/robot1/robot/ground_truth`
   
   **方法B - 手动选择**：
   - 在话题列表中勾选感兴趣的话题
   - 需要通过代码扩展支持自定义话题的数据提取

4. **查看数据摘要**
   - 提取成功后，右侧信息区显示该话题的详细统计数据：
     - 数据点数量
     - 时间范围和持续时间
     - X/Y/Z坐标范围
     - Roll/Pitch/Yaw角度范围

5. **生成可视化图表**

   点击左侧快速操作区域的按钮：
   
   - **📈 绘制位置曲线**：X、Y、Z坐标随时间的变化曲线
   - **📊 绘制姿态曲线**：Roll、Pitch、Yaw角度随时间的变化曲线  
   - **🗺️ 绘制2D轨迹**：XY平面的运动轨迹图（含起点终点标记）
   - **🎛️ 综合仪表板**：多图表组合视图（位置、姿态、轨迹、统计）

6. **应用滤波处理**

   a. **配置滤波参数**（在"🔧 滤波设置"区域）：
      - 选择滤波类型（下拉菜单）
      - 调整对应参数：
        - *窗口大小*：滑动窗口的点数（1-51）
        - *平滑系数α*：指数平滑的权重因子（0.01-1.0）
        - *截止频率*：低通滤波器的截止频率（Hz）
        - *过程噪声Q*：卡尔曼滤波的系统噪声参数
        - *测量噪声R*：卡尔曼滤波的传感器噪声参数
   
   b. **执行滤波**：
      - 点击"✨ 应用滤波"按钮
      - 程序自动对所有已提取的话题数据应用当前设置的滤波器
   
   c. **查看效果**：
      - 切换到"🔍 滤波对比"标签页
      - 查看每个字段的详细统计对比（标准差降低百分比、RMSE、相关性等）
      - 重新绘制图表时，会同时显示原始数据（实线）和滤波后数据（虚线）

7. **导出结果**

   - **导出数据**：点击"💾 导出数据(CSV)"将所有话题数据导出为CSV文件
   - **保存图表**：点击"🖼️ 保存所有图表"将生成的图表保存为PNG/PDF等格式

### 方式二：命令行/脚本方式

适合批量处理或自动化场景：

```python
from bag_parser import BagParser
from data_visualizer import DataVisualizer
from filter_processor import FilterProcessor, FilterConfig

# 1. 解析bag文件
parser = BagParser('your_data.bag')
topics = parser.parse_bag()

# 2. 打印统计信息
print(parser.get_statistics_report())

# 3. 提取特定话题的位姿数据
pose_data = parser.extract_pose_data('/Odometry')

# 4. 创建可视化器并添加数据
visualizer = DataVisualizer()
visualizer.add_pose_data('/Odometry', pose_data)

# 5. 绘制图表
fig_pos = visualizer.plot_position_time(save_path='position.png')
fig_ori = visualizer.plot_orientation_time(save_path='orientation.png')
fig_traj = visualizer.plot_2d_trajectory(save_path='trajectory.png')

# 6. 应用滤波
processor = FilterProcessor()
config = FilterConfig(filter_type='kalman', process_noise=0.01, measurement_noise=0.1)
filtered_pose = processor.apply_filter_to_pose_data(pose_data, config)

# 7. 添加滤波后数据并对比
visualizer.add_pose_data('/Odometry', filtered_pose, 'filtered')
fig_compare = visualizer.plot_comparison('/Odometry', 'x', save_path='comparison.png')

# 8. 显示所有图表
visualizer.show_all_plots()
```

---

## 📚 功能模块详细介绍

### 1. 数据提取模块 (`bag_parser.py`)

#### 核心类：`BagParser`

**主要方法**：

| 方法 | 功能 | 返回值 |
|------|------|--------|
| `parse_bag()` | 解析整个bag文件 | Dict[str, TopicInfo] |
| `get_all_topics()` | 获取所有话题信息列表 | List[TopicInfo] |
| `extract_pose_data(topic_name)` | 提取指定位姿数据 | PoseData对象 |
| `export_topics_to_csv(dir)` | 导出数据到CSV | bool |
| `get_statistics_report()` | 生成文本统计报告 | str |

**支持的ROS消息类型**：
- `nav_msgs/Odometry` - 里程计数据
- `geometry_msgs/PoseStamped` - 位姿消息
- `nav_msgs/Path` - 路径消息（提取第一个点作为示例）

**数据结构**：

```python
@dataclass
class TopicInfo:
    name: str           # 话题名称
    msg_type: str       # 消息类型
    message_count: int  # 消息数量
    start_time: float   # 开始时间戳(s)
    end_time: float     # 结束时间戳(s)
    duration: float     # 持续时间(s)

@dataclass
class PoseData:
    timestamp: np.ndarray      # 时间戳数组
    x, y, z: np.ndarray        # 位置坐标(m)
    roll, pitch, yaw: np.ndarray  # 姿态角(rad)
    quaternion_w/x/y/z: np.ndarray  # 四元数分量
```

### 2. 数据可视化模块 (`data_visualizer.py`)

#### 核心类：`DataVisualizer`

**绘图函数**：

| 函数名 | 功能描述 | 输出 |
|--------|----------|------|
| `plot_position_time()` | X/Y/Z随时间变化曲线 | 3×1子图 |
| `plot_orientation_time()` | Roll/Pitch/Yaw随时间变化曲线 | 3×1子图 |
| `plot_2d_trajectory()` | XY平面轨迹投影 | 单图 |
| `plot_3d_trajectory()` | 三维空间轨迹 | 3D交互图 |
| `plot_comparison()` | 单字段滤波前后对比 + 残差 | 2×1子图 |
| `create_dashboard()` | 综合仪表板 | 2×2子图布局 |

**特性**：
- 自动检测并显示中文字体
- 支持同时显示原始数据和滤波后数据
- 可自定义图表尺寸、颜色、保存格式
- 所有图表包含网格线、图例、轴标签

### 3. 滤波处理模块 (`filter_processor.py`)

#### 核心类：`FilterProcessor`

**支持的7种滤波算法**：

#### ① 滑动平均滤波 (Moving Average)
```
适用场景：高频随机噪声去除
原理：y[i] = mean(x[i-k]) for k in window
特点：简单高效，但引入相位延迟
参数：window_size (窗口大小，建议3-15)
```

#### ② 加权滑动平均 (Weighted Moving Average)
```
适用场景：需保留信号局部特征
原理：三角权重，中心点权重最高
特点：减少相位延迟，边缘保持更好
参数：window_size
```

#### ③ 指数加权移动平均 (Exponential MA)
```
适用场景：实时数据流平滑
原理：y[i] = α*x[i] + (1-α)*y[i-1]
特点：无需窗口大小，计算效率高
参数：alpha (0.01-1.0，越小越平滑)
```

#### ④ 中值滤波 (Median Filter)
```
适用场景：脉冲噪声（椒盐噪声）去除
原理：用窗口内中位数替代中心值
特点：保持边缘锐利，对高斯噪声效果一般
参数：window_size (建议奇数3-9)
```

#### ⑤ 卡尔曼滤波 (Kalman Filter)
```
适用场景：高斯白噪声下的最优估计
原理：基于状态空间的递归最优估计
特点：理论最优（均方误差最小），需调参
参数：process_noise (过程噪声Q), measurement_noise (测量噪声R)
调参技巧：
  - 测量噪声大 → 增大R
  - 系统变化快 → 增大Q
```

#### ⑥ 巴特沃斯低通滤波 (Butterworth Low-pass)
```
适用场景：滤除特定频率以上的成分
原理：IIR滤波器，通带最大平坦幅频响应
特点：频率选择性明确，可能产生振铃效应
参数：cutoff_freq (截止频率Hz), order (阶数2-6)
```

#### ⑦ Savitzky-Golay滤波
```
适用场景：光谱数据、需保留峰值特征
原理：局部多项式拟合平滑
特点：保留高阶矩，峰值高度宽度不变
参数：window_size (多项式阶数自动设为window//4)
```

**其他实用功能**：

| 方法 | 功能 |
|------|------|
| `remove_outliers()` | 异常值检测剔除（IQR/Z-Score等方法） |
| `interpolate_missing()` | 缺失值插值填补（线性/三次/最近邻） |
| `modify_data_point()` | 手动修改单个数据点 |
| `get_filter_comparison_stats()` | 计算滤波效果统计指标 |
| `print_filter_report()` | 生成详细的滤波效果文本报告 |

**统计指标说明**：
- **标准差降低%**：衡量平滑程度，越高越平滑
- **RMSE**：原始与滤波数据的均方根误差
- **相关性**：衡量信号形状保留程度（1.0=完全保留）
- **SNR改善(dB)**：估算的信噪比提升分贝数

### 4. 主程序 (`main.py`)

#### GUI界面结构

```
┌─────────────────────────────────────────────────────┐
│  菜单栏: 文件 | 视图 | 帮助                           │
├──────────────┬──────────────────────────────────────┤
│              │                                      │
│  左侧控制面板  │         右侧显示区域                  │
│              │                                      │
│ ┌──────────┐ │  ┌─────────────────────────────────┐│
│ │ 文件操作  │ │  │  [信息] [图表] [滤波对比] 标签页   ││
│ └──────────┘ │  │                                 ││
│              │  │                                 ││
│ ┌──────────┐ │  │     数据/图表/报告显示区域          ││
│ │ 话题列表  │ │  │                                 ││
│ └──────────┘ │  │                                 ││
│              │  │                                 ││
│ ┌──────────┐ │  │                                 ││
│ │目标话题   │ │  │                                 ││
│ └──────────┘ │  │                                 ││
│              │  └─────────────────────────────────┘│
│ ┌──────────┐ │                                      │
│ │ 滤波设置  │ │                                      │
│ └──────────┘ │                                      │
│              │                                      │
│ ┌──────────┐ │                                      │
│ │ 快速操作  │ │                                      │
│ └──────────┘ │                                      │
│              │                                      │
├──────────────┴──────────────────────────────────────┤
│  状态栏: 就绪 | 进度条                                │
└─────────────────────────────────────────────────────┘
```

**关键组件**：
- **后台线程加载**：避免UI卡顿，大文件也能流畅操作
- **多选话题支持**：可同时分析多个话题并进行对比
- **实时反馈**：状态栏显示当前操作进度和结果
- **快捷键支持**：Ctrl+O打开文件，Ctrl+E导出，Ctrl+S保存

---

## 🔧 参数调优指南

### 滤波器选择建议

| 数据特征 | 推荐滤波器 | 典型参数 |
|----------|-----------|----------|
| 高频白噪声 | Moving Average | window=5-11 |
| 缓慢趋势+噪声 | Exponential MA | alpha=0.2-0.4 |
| 脉冲干扰 | Median | window=3-7 |
| 高精度传感器 | Kalman | Q=0.001-0.01, R=0.05-0.5 |
| 已知噪声频率 | Butterworth | cutoff=噪声频率×1.5 |
| 光谱/波形数据 | Savitzky-Golay | window=11-21 |
| 需要实时处理 | Exponential MA | alpha=0.3-0.5 |

### 性能优化建议

1. **大型bag文件 (>100MB)**：
   - 先使用`parse_bag()`获取话题概览
   - 只提取需要的话题数据
   - 考虑降采样后再可视化

2. **大量数据点 (>10000)**：
   - 使用较小的滤波窗口
   - 优先选择EMA或卡尔曼滤波（O(n)复杂度）
   - 分段处理和绘图

3. **内存不足时**：
   - 关闭不需要的图表窗口
   - 定期调用`visualizer.close_all_figures()`
   - 减少同时加载的话题数量

---

## ❓ 常见问题解决

### Q1: 安装rosbag失败

**问题现象**：
```
ERROR: Could not find a version that satisfies the requirement rosbag
```

**解决方案**：
- **Windows/macOS系统**：尝试安装纯Python版本
  ```bash
  pip install rosbag  # 如果ROS未安装
  ```
- **Linux系统**：确保已安装ROS并source环境
  ```bash
  source /opt/ros/<distro>/setup.bash
  pip install rosbag
  ```

### Q2: 中文显示乱码或方框

**问题现象**：图表中的中文标题显示为方框或乱码

**解决方案**：
1. 安装中文字体（如SimHei、Microsoft YaHei）
2. 在代码开头添加字体配置：
   ```python
   import matplotlib.pyplot as plt
   plt.rcParams['font.sans-serif'] = ['SimHei']
   plt.rcParams['axes.unicode_minus'] = False
   ```
3. 或修改`data_visualizer.py`中的字体设置

### Q3: 无法打开某些.bag文件

**可能原因及解决**：
- **文件损坏**：检查文件完整性，尝试在其他工具中打开
- **ROS版本不匹配**：确认bag文件是ROS1格式（非ROS2）
- **权限问题**：确保有文件的读取权限
- **路径问题**：使用绝对路径而非相对路径

### Q4: 某些话题无法提取位姿数据

**原因**：
- 该话题的消息类型不包含位姿字段（如`cmd_vel`是速度指令）
- 消息结构与预期不符

**解决方法**：
- 查看该话题的实际消息类型（在信息面板中可见）
- 如需支持新类型，需扩展`bag_parser.py`中的`_extract_pose_from_message()`方法

### Q5: 滤波后数据出现相位延迟

**现象**：滤波后的曲线相对于原始数据有时间偏移

**原因**：因果滤波器（如MA、EMA）固有的延迟特性

**缓解方法**：
- 使用更小的窗口大小
- 尝试零相位滤波（如巴特沃斯的filtfilt模式）
- 对非实时应用，可考虑Savitzky-Golay滤波

### Q6: GUI界面无响应

**原因**：正在处理大型数据或复杂计算

**解决**：
- 等待几秒钟（程序使用多线程，通常不会长时间卡死）
- 检查控制台是否有错误输出
- 减少数据量后重试
- 任务管理器中确认进程仍在运行

### Q7: 图表保存失败或空白

**排查步骤**：
1. 确保保存目录存在且有写入权限
2. 检查磁盘空间是否充足
3. 尝试不同的图片格式（PNG/PDF/SVG）
4. 降低DPI设置（默认300，可改为150）

### Q8: 如何在没有ROS的机器上运行？

**方案**：安装独立的rosbag库
```bash
pip install roscpp_numpy empy osqp catkin_pkg
pip install rosbag
```

**限制**：只能读取已录制好的.bag文件，无法连接实时ROS系统

---

## 📊 示例数据与运行结果

### 示例场景

假设有一个机器人导航实验的bag文件，包含以下话题：
- `/Odometry` - 里程计数据（2000条消息）
- `/path` - 规划路径（500条消息）
- `/robot1/robot/cmd_vel` - 速度指令（3000条消息）
- `/robot1/robot/ground_truth` - 真实位姿（2000条消息）

### 预期输出示例

#### 1. 位置-时间曲线图 (position_time.png)
![位置曲线示例](docs/example_position.png)

- 显示X/Y/Z三个坐标随时间变化
- 不同颜色代表不同话题
- 实线=原始数据，虚线=滤波后数据
- 包含网格线和清晰图例

#### 2. 姿态-时间曲线图 (orientation_time.png)
![姿态曲线示例](docs/example_orientation.png)

- Roll/Pitch/Yaw角度以度为单位显示
- Yaw角通常呈现单调变化（旋转累积）
- Roll和Pitch在小范围内波动

#### 3. 2D轨迹图 (trajectory_2d.png)
![2D轨迹示例](docs/example_trajectory_2d.png)

- XY平面投影的运动轨迹
- 圆形标记起点，方形标记终点
- 可观察机器人运动模式和覆盖范围

#### 4. 滤波对比报告示例

```
============================================================
滤波器: 卡尔曼滤波
============================================================

话题: /Odometry
----------------------------------------

  字段: X
    标准差降低: 45.23%
    RMSE: 0.023456
    相关性: 0.9876

  字段: Y
    标准差降低: 42.87%
    RMSE: 0.031234
    相关性: 0.9854

  字段: Z
    标准差降低: 38.56%
    RMSE: 0.008765
    相关性: 0.9923
```

### 快速测试代码

如果没有实际的bag文件，可以使用模拟数据进行测试：

```python
import numpy as np
from bag_parser import PoseData
from data_visualizer import DataVisualizer
from filter_processor import FilterProcessor, FilterConfig

# 生成模拟数据
n = 200
t = np.linspace(0, 10, n)

test_pose = PoseData(
    timestamp=t,
    x=np.sin(t) + np.random.normal(0, 0.05, n),
    y=np.cos(t) + np.random.normal(0, 0.05, n),
    z=np.linspace(0, 2, n),
    roll=np.random.normal(0, 0.02, n),
    pitch=np.random.normal(0, 0.02, n),
    yaw=t * 0.3,
    quaternion_w=np.ones(n),
    quaternion_x=np.zeros(n),
    quaternion_y=np.zeros(n),
    quaternion_z=np.zeros(n)
)

# 可视化
viz = DataVisualizer()
viz.add_pose_data('/simulated_robot', test_pose)

# 绘制各种图表
viz.plot_position_time(save_path='test_position.png')
viz.plot_orientation_time(save_path='test_orientation.png')
viz.plot_2d_trajectory(save_path='test_trajectory.png')

# 应用滤波测试
processor = FilterProcessor()
config = FilterConfig(filter_type='moving_average', window_size=7)
filtered = processor.apply_filter_to_pose_data(test_pose, config)

viz.add_pose_data('/simulated_robot', filtered, 'filtered')
viz.create_dashboard(save_path='test_dashboard.png')

print("测试完成！请查看 ./ 目录下生成的图片文件")
```

运行上述代码将生成以下文件（无需真实bag文件）：
- `test_position.png` - 位置曲线
- `test_orientation.png` - 姿态曲线
- `test_trajectory.png` - 2D轨迹
- `test_dashboard.png` - 综合仪表板

---

## 📁 项目文件结构

```
Demo1/
├── main.py                 # 主程序入口（GUI界面）
├── bag_parser.py           # Bag文件解析模块
├── data_visualizer.py      # 数据可视化模块
├── filter_processor.py     # 滤波处理模块
├── requirements.txt        # 依赖库清单
├── README.md               # 本文档
│
├── exported_data/          # CSV导出目录（运行后生成）
│   ├── topics_summary.csv
│   └── <topic>_data.csv
│
├── plots/                  # 图表保存目录（运行后生成）
│   ├── position_time.png
│   ├── orientation_time.png
│   ├── trajectory_2d.png
│   └── ...
│
└── temp_plots/             # 临时图表缓存
```

---

## 🔬 扩展开发指南

### 添加新的滤波算法

在`filter_processor.py`的`FilterProcessor`类中：

1. 添加新的滤波方法（以`_`开头的私有方法）
2. 在`apply_filter()`方法的`filter_methods`字典中注册
3. 在`available_filters`列表中添加名称
4. 在GUI的`main.py`中更新下拉菜单选项

**示例**：添加高斯滤波

```python
def _gaussian_filter(self, data, config):
    from scipy.ndimage import gaussian_filter1d
    sigma = config.window_size / 3.0  # 将窗口大小转换为sigma
    return gaussian_filter1d(data, sigma=sigma, mode='reflect')
```

### 支持新的ROS消息类型

在`bag_parser.py`的`BagParser._extract_pose_from_message()`方法中添加对新消息结构的解析逻辑：

```python
elif hasattr(msg, 'your_custom_field'):
    # 解析自定义消息结构
    position = (msg.x, msg.y, msg.z)
    quaternion = (msg.qw, msg.qx, msg.qy, msg.qz)
    return (position, quaternion)
```

### 自定义可视化样式

通过修改`data_visualizer.py`中的绘图函数，可以自定义：
- 颜色方案（`self.colors`列表）
- 线型（`self.line_styles`列表）
- 图表尺寸（`figsize`参数）
- 字体和字号

---

## 📄 许可证

本项目仅供学习和研究使用。

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进本项目！

### 开发规范
- 遵循PEP 8代码风格
- 为新增功能提供完整注释
- 更新本文档的相关章节
- 确保向后兼容性

---

## 📞 技术支持

如有问题，请：
1. 查看"常见问题解决"章节
2. 检查GitHub Issues是否已有类似问题
3. 提交新的Issue并提供：
   - 操作系统和Python版本
   - 完整的错误信息截图
   - 复现步骤
   - bag文件的基本信息（大小、话题数等）

---

## ⚠️ 常见问题快速解决

### 问题：遇到 "utf-8 codec can't decode byte" 错误

**症状**：
```
解析消息时出错: 'utf-8' codec can't decode byte 0xc2 in position 11
```

**原因**：rosbags库在解析某些消息类型时遇到编码问题

**解决方案（按推荐顺序）**：

#### ✅ 方案1：运行诊断工具（立即尝试）
```bash
python diagnose.py your_file.bag
```
查看每个话题的成功率统计

#### ✅ 方案2：安装完整依赖包
```bash
pip install rosbags[all]
```

#### ✅ 方案3：使用Docker容器（最稳定）
详见 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

#### ✅ 方案4：使用传统ROS环境（如果有条件）
```bash
# 在Ubuntu + ROS环境中
parser = BagParser('file.bag', backend='rosbag')
```

**注意**：程序已优化错误处理，会自动跳过无法解析的消息并显示成功率。如果成功率 > 70%，分析结果仍然有效！

**详细文档**：请查看 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) 获取完整的故障排除指南。

---

## 📝 版本历史

### v1.0.0 (2026-04-27)
- ✅ 初始版本发布
- ✅ 完整的bag文件解析功能
- ✅ 7种滤波算法实现
- ✅ GUI界面和命令行双模式
- ✅ 数据导出和可视化功能
- ✅ 详细的使用文档

---

**祝您使用愉快！如有任何问题，欢迎随时反馈。** 🎉
