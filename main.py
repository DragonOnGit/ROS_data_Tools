# -*- coding: utf-8 -*-
"""
ROS Bag文件数据处理与可视化系统 - 主程序
功能：提供图形用户界面，整合所有功能模块
版本：3.4.0 - 管道范围 + 数据显示模式
"""

import sys
import os
import io
import numpy as np
from typing import Optional
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QListWidget, QTextEdit, QFileDialog, QComboBox,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
                             QMessageBox, QSplitter, QProgressBar, QStatusBar,
                             QAction, QMenu, QMenuBar, QDialog, QLineEdit,
                             QScrollArea, QCheckBox, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QSettings
from PyQt5.QtGui import QIcon, QFont, QTextCursor

from bag_parser import BagParser, TopicInfo, PoseData
from data_visualizer import DataVisualizer
from filter_processor import FilterProcessor, FilterConfig
from octree_map import OctreeMap, OctreeConfig
from octree_visualizer import OctreeVisualizer
from occupancy_grid import OccupancyGridMap, OccupancyGridConfig, CoordinateTransformer, TrajectoryOverlay

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

_font_candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                    'Noto Sans CJK SC', 'PingFang SC', 'Heiti SC',
                    'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = _font_candidates
plt.rcParams['axes.unicode_minus'] = False

POSE_MSG_TYPES = {
    'nav_msgs/Odometry', 'geometry_msgs/PoseStamped',
    'geometry_msgs/PoseWithCovarianceStamped', 'nav_msgs/Path',
    'geometry_msgs/Twist', 'geometry_msgs/TwistStamped',
    'geometry_msgs/Vector3Stamped', 'sensor_msgs/Imu',
}
MAPPING_MSG_TYPES = {'sensor_msgs/PointCloud2'}


class ConsoleSignaler(QObject):
    text_written = pyqtSignal(str)


class ConsoleStream(io.TextIOBase):
    def __init__(self, signaler: ConsoleSignaler, tag: str = ''):
        super().__init__()
        self.signaler = signaler
        self.tag = tag
        self._original = None

    def write(self, text: str):
        if text and text.strip():
            self.signaler.text_written.emit(text)
        return len(text) if text else 0

    def flush(self):
        pass


class BagLoadingThread(QThread):
    loading_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(self, bag_path: str):
        super().__init__()
        self.bag_path = bag_path

    def run(self):
        try:
            self.progress_update.emit("正在初始化解析器...")
            parser = BagParser(self.bag_path)
            self.progress_update.emit("正在读取bag文件...")
            parser.parse_bag()
            self.loading_finished.emit(parser)
        except Exception as e:
            self.error_occurred.emit(f"加载失败: {str(e)}")


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.parser = None
        self.visualizer = DataVisualizer()
        self.filter_processor = FilterProcessor()
        self.current_pose_data = {}
        self.filtered_pose_data = {}
        self.octree_map = None
        self.octree_visualizer = None
        self.global_ogm = None
        self.local_ogm = None
        self._drone_altitude = None
        self.coord_transformer = CoordinateTransformer(flip_x=True, flip_y=True)
        self._actual_traj_xy = None
        self._expected_traj_xy = None
        self._gt_time_range = (0.0, 0.0)
        self._gt_timestamps = None
        self._pipe_upper_data = None
        self._pipe_lower_data = None

        self.plot_canvas = None
        self._plot_toolbar = None
        self.canvas_3d = None
        self._toolbar_3d = None
        self.canvas_octree = None
        self._toolbar_octree = None
        self.canvas_ogm = None
        self._toolbar_ogm = None
        self.filter_canvas = None
        self._filter_tb = None

        self._topic_buttons_pose = []
        self._topic_buttons_map = []

        self.setWindowTitle("ROS Bag数据分析与可视化系统")
        self.setGeometry(100, 100, 1500, 900)
        self.setMinimumSize(1200, 700)
        self._setup_ui()
        self._create_menu_bar()
        self._create_status_bar()
        self._setup_console_redirect()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([380, 1120])
        layout.addWidget(splitter)

    # ==================== Left Panel ====================

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.left_tabs = QTabWidget()
        self.left_tabs.setTabPosition(QTabWidget.West)

        self.left_tabs.addTab(self._create_load_tab(), "加载数据")
        self.left_tabs.addTab(self._create_pose_tab(), "位姿数据处理")
        self.left_tabs.addTab(self._create_mapping_tab(), "建图处理")

        layout.addWidget(self.left_tabs)
        return panel

    def _create_load_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(6)

        fg = QGroupBox("📁 文件操作")
        fl = QVBoxLayout(fg)
        btn = QPushButton("📂 打开Bag文件")
        btn.setMinimumHeight(35)
        btn.clicked.connect(self.open_bag_file)
        self.lbl_file = QLabel("未选择文件")
        self.lbl_file.setWordWrap(True)
        fl.addWidget(btn)
        fl.addWidget(self.lbl_file)
        lay.addWidget(fg)

        tg = QGroupBox("📋 话题列表")
        tl = QVBoxLayout(tg)
        self.list_topics = QListWidget()
        self.list_topics.setSelectionMode(QListWidget.MultiSelection)
        tl.addWidget(self.list_topics)
        lay.addWidget(tg)

        ig = QGroupBox("ℹ️ INFO日志")
        il = QVBoxLayout(ig)
        self.text_console = QTextEdit()
        self.text_console.setReadOnly(True)
        self.text_console.setFont(QFont("Consolas", 8))
        self.text_console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        il.addWidget(self.text_console)
        lay.addWidget(ig)
        return tab

    def _create_pose_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setSpacing(6)

        tg = QGroupBox("📋 位姿话题提取")
        tl = QVBoxLayout(tg)
        self.pose_btn_container = QWidget()
        self.pose_btn_layout = QVBoxLayout(self.pose_btn_container)
        self.pose_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.pose_btn_layout.setSpacing(3)
        lbl = QLabel("请先加载bag文件")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #999; font-style: italic;")
        self.pose_btn_layout.addWidget(lbl)
        tl.addWidget(self.pose_btn_container)
        lay.addWidget(tg)

        pipe_g = QGroupBox("📏 管道范围设置")
        pipe_l = QFormLayout(pipe_g)
        self.combo_pipe_upper = QComboBox()
        self.combo_pipe_upper.addItem("未选择", None)
        self.combo_pipe_upper.currentIndexChanged.connect(self._on_pipe_topic_changed)
        pipe_l.addRow("管道上界话题:", self.combo_pipe_upper)
        self.combo_pipe_lower = QComboBox()
        self.combo_pipe_lower.addItem("未选择", None)
        self.combo_pipe_lower.currentIndexChanged.connect(self._on_pipe_topic_changed)
        pipe_l.addRow("管道下界话题:", self.combo_pipe_lower)
        self.lbl_pipe_status = QLabel("未选择管道话题")
        self.lbl_pipe_status.setStyleSheet("color: #999; font-size: 9px;")
        pipe_l.addRow(self.lbl_pipe_status)
        self.dspin_pipe_alpha = QDoubleSpinBox(); self.dspin_pipe_alpha.setRange(0.05, 1.0)
        self.dspin_pipe_alpha.setValue(0.3); self.dspin_pipe_alpha.setSingleStep(0.05)
        pipe_l.addRow("管道透明度:", self.dspin_pipe_alpha)
        self.spin_pipe_lw = QDoubleSpinBox(); self.spin_pipe_lw.setRange(0.5, 5.0)
        self.spin_pipe_lw.setValue(1.0); self.spin_pipe_lw.setSingleStep(0.5)
        pipe_l.addRow("管道线宽:", self.spin_pipe_lw)
        self.chk_pipe_show = QCheckBox("3D轨迹中显示管道范围")
        self.chk_pipe_show.setChecked(True)
        pipe_l.addRow(self.chk_pipe_show)
        lay.addWidget(pipe_g)

        fg = QGroupBox("🔧 滤波设置")
        fl = QFormLayout(fg)
        self.combo_filter = QComboBox()
        self.combo_filter.addItems([
            '滑动平均', '加权滑动平均', '指数平滑',
            '中值滤波', '卡尔曼滤波', '巴特沃斯低通', 'Savitzky-Golay'
        ])
        fl.addRow("滤波类型:", self.combo_filter)
        self.spin_win = QSpinBox(); self.spin_win.setRange(1, 51); self.spin_win.setValue(7)
        fl.addRow("窗口大小:", self.spin_win)
        self.dspin_alpha = QDoubleSpinBox(); self.dspin_alpha.setRange(0.01, 1.0)
        self.dspin_alpha.setValue(0.3); self.dspin_alpha.setSingleStep(0.05)
        fl.addRow("平滑系数 α:", self.dspin_alpha)
        self.dspin_cutoff = QDoubleSpinBox(); self.dspin_cutoff.setRange(0.1, 50.0)
        self.dspin_cutoff.setValue(5.0)
        fl.addRow("截止频率(Hz):", self.dspin_cutoff)
        self.dspin_q = QDoubleSpinBox(); self.dspin_q.setRange(0.001, 1.0)
        self.dspin_q.setValue(0.01); self.dspin_q.setDecimals(4); self.dspin_q.setSingleStep(0.005)
        fl.addRow("过程噪声 Q:", self.dspin_q)
        self.dspin_r = QDoubleSpinBox(); self.dspin_r.setRange(0.001, 10.0)
        self.dspin_r.setValue(0.1); self.dspin_r.setDecimals(3)
        fl.addRow("测量噪声 R:", self.dspin_r)
        btn_f = QPushButton("✨ 应用滤波"); btn_f.setMinimumHeight(35)
        btn_f.clicked.connect(self.apply_filter)
        fl.addRow(btn_f)
        lay.addWidget(fg)

        pg = QGroupBox("📊 绘图功能")
        pl = QVBoxLayout(pg)
        dm_l = QHBoxLayout()
        dm_l.addWidget(QLabel("数据显示:"))
        self.combo_display_mode = QComboBox()
        self.combo_display_mode.addItems(['显示滤波数据', '显示原始数据', '同时显示两者'])
        self.combo_display_mode.setCurrentIndex(2)
        dm_l.addWidget(self.combo_display_mode)
        pl.addLayout(dm_l)
        for text, slot in [
            ("📈 绘制位置曲线", self.plot_position),
            ("📊 绘制姿态曲线", self.plot_orientation),
            ("🗺️ 绘制2D轨迹", self.plot_2d),
            ("🌐 绘制3D轨迹", self.plot_3d),
            ("🎛️ 综合仪表板", self.plot_dashboard),
        ]:
            b = QPushButton(text); b.clicked.connect(slot); pl.addWidget(b)
        btn_clr = QPushButton("🗑️ 清除曲线")
        btn_clr.setStyleSheet(
            "QPushButton{background:#e74c3c;color:white;border-radius:4px;font-weight:bold}"
            "QPushButton:hover{background:#c0392b}")
        btn_clr.clicked.connect(self.clear_all_curves)
        pl.addWidget(btn_clr)
        lay.addWidget(pg)

        eg = QGroupBox("💾 数据导出")
        el = QVBoxLayout(eg)
        b1 = QPushButton("💾 导出数据(CSV)"); b1.clicked.connect(self.export_data); el.addWidget(b1)
        b2 = QPushButton("🖼️ 保存所有图表"); b2.clicked.connect(self.save_plots); el.addWidget(b2)
        lay.addWidget(eg)

        lay.addStretch()
        scroll.setWidget(content)
        return scroll

    def _create_mapping_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setSpacing(6)

        tg = QGroupBox("📡 点云话题选择")
        tl = QVBoxLayout(tg)
        self.map_btn_container = QWidget()
        self.map_btn_layout = QVBoxLayout(self.map_btn_container)
        self.map_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.map_btn_layout.setSpacing(3)
        lbl = QLabel("请先加载bag文件")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #999; font-style: italic;")
        self.map_btn_layout.addWidget(lbl)
        tl.addWidget(self.map_btn_container)
        lay.addWidget(tg)

        og = QGroupBox("🗺️ 八叉树地图")
        ol = QFormLayout(og)
        self.spin_voxel = QDoubleSpinBox(); self.spin_voxel.setRange(0.01, 10.0)
        self.spin_voxel.setValue(0.1); self.spin_voxel.setSingleStep(0.05); self.spin_voxel.setDecimals(3)
        ol.addRow("体素大小(m):", self.spin_voxel)
        self.spin_maxd = QSpinBox(); self.spin_maxd.setRange(1, 16); self.spin_maxd.setValue(8)
        ol.addRow("最大深度:", self.spin_maxd)
        self.spin_mind = QSpinBox(); self.spin_mind.setRange(0, 16); self.spin_mind.setValue(0)
        ol.addRow("最小深度:", self.spin_mind)
        self.spin_dthresh = QSpinBox(); self.spin_dthresh.setRange(1, 1000); self.spin_dthresh.setValue(5)
        ol.addRow("密度阈值:", self.spin_dthresh)
        self.spin_hmin = QDoubleSpinBox(); self.spin_hmin.setRange(-1000, 1000)
        self.spin_hmin.setValue(-999); self.spin_hmin.setDecimals(2)
        ol.addRow("高度下限(m):", self.spin_hmin)
        self.spin_hmax = QDoubleSpinBox(); self.spin_hmax.setRange(-1000, 1000)
        self.spin_hmax.setValue(999); self.spin_hmax.setDecimals(2)
        ol.addRow("高度上限(m):", self.spin_hmax)
        self.combo_cscheme = QComboBox()
        self.combo_cscheme.addItems(['高度映射', '密度映射', '深度映射', '自定义'])
        ol.addRow("颜色方案:", self.combo_cscheme)
        self.dspin_transp = QDoubleSpinBox(); self.dspin_transp.setRange(0.0, 1.0)
        self.dspin_transp.setValue(0.8); self.dspin_transp.setSingleStep(0.05)
        ol.addRow("透明度:", self.dspin_transp)
        self.combo_rqual = QComboBox(); self.combo_rqual.addItems(['低', '中', '高'])
        self.combo_rqual.setCurrentIndex(1)
        ol.addRow("渲染质量:", self.combo_rqual)
        self.chk_borders = QCheckBox("显示边界线")
        ol.addRow(self.chk_borders)
        btn_oct = QPushButton("🏗️ 生成八叉树地图")
        btn_oct.setMinimumHeight(35)
        btn_oct.setStyleSheet(
            "QPushButton{background:#2196F3;color:white;border-radius:4px;font-weight:bold}"
            "QPushButton:hover{background:#1976D2}")
        btn_oct.clicked.connect(self.build_octree)
        ol.addRow(btn_oct)
        btn_oct2d = QPushButton("🗺️ 2D俯视投影"); btn_oct2d.clicked.connect(self.octree_2d); ol.addRow(btn_oct2d)
        btn_octst = QPushButton("📊 统计分析"); btn_octst.clicked.connect(self.octree_stats); ol.addRow(btn_octst)
        lay.addWidget(og)

        gg = QGroupBox("🧱 占据网格图")
        gl = QFormLayout(gg)
        self.spin_ogm_res = QDoubleSpinBox(); self.spin_ogm_res.setRange(0.01, 5.0)
        self.spin_ogm_res.setValue(0.05); self.spin_ogm_res.setSingleStep(0.01); self.spin_ogm_res.setDecimals(3)
        gl.addRow("分辨率(m/格):", self.spin_ogm_res)
        self.spin_ogm_hmin = QDoubleSpinBox(); self.spin_ogm_hmin.setRange(-1000, 1000)
        self.spin_ogm_hmin.setValue(-999); self.spin_ogm_hmin.setDecimals(2)
        gl.addRow("高度下限(m):", self.spin_ogm_hmin)
        self.spin_ogm_hmax = QDoubleSpinBox(); self.spin_ogm_hmax.setRange(-1000, 1000)
        self.spin_ogm_hmax.setValue(999); self.spin_ogm_hmax.setDecimals(2)
        gl.addRow("高度上限(m):", self.spin_ogm_hmax)
        self.spin_ogm_occ = QSpinBox(); self.spin_ogm_occ.setRange(1, 1000); self.spin_ogm_occ.setValue(3)
        gl.addRow("占据阈值:", self.spin_ogm_occ)
        btn_global = QPushButton("🌍 生成全局占据网格图")
        btn_global.setMinimumHeight(32)
        btn_global.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;border-radius:4px;font-weight:bold}"
            "QPushButton:hover{background:#388E3C}")
        btn_global.clicked.connect(self.build_global_ogm)
        gl.addRow(btn_global)

        sep = QLabel("── 局部占据网格图参数 ──")
        sep.setAlignment(Qt.AlignCenter)
        sep.setStyleSheet("color: #888; font-size: 10px;")
        gl.addRow(sep)
        self.spin_local_range = QDoubleSpinBox(); self.spin_local_range.setRange(1.0, 1000.0)
        self.spin_local_range.setValue(10.0); self.spin_local_range.setSingleStep(1.0)
        gl.addRow("局部范围(m):", self.spin_local_range)
        
        time_sep = QLabel("── 时间点选择 ──")
        time_sep.setAlignment(Qt.AlignCenter)
        time_sep.setStyleSheet("color: #888; font-size: 10px;")
        gl.addRow(time_sep)
        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setRange(0, 1000)
        self.slider_time.setValue(0)
        self.slider_time.setTickPosition(QSlider.TicksBelow)
        self.slider_time.setTickInterval(100)
        self.slider_time.valueChanged.connect(self._on_time_slider_changed)
        gl.addRow("时间滑块:", self.slider_time)
        self.spin_time_point = QDoubleSpinBox(); self.spin_time_point.setRange(0, 99999)
        self.spin_time_point.setValue(0); self.spin_time_point.setDecimals(2)
        self.spin_time_point.setSingleStep(0.1)
        self.spin_time_point.valueChanged.connect(self._on_time_spin_changed)
        gl.addRow("时间点(s):", self.spin_time_point)
        self.lbl_time_range = QLabel("未加载")
        self.lbl_time_range.setStyleSheet("color: #888; font-size: 9px;")
        gl.addRow("数据时间范围:", self.lbl_time_range)
        self.lbl_gt_pos = QLabel("未获取")
        self.lbl_gt_pos.setStyleSheet("color: #666;")
        gl.addRow("无人机位置:", self.lbl_gt_pos)
        self.lbl_gt_alt = QLabel("未获取")
        self.lbl_gt_alt.setStyleSheet("color: #1565C0; font-weight: bold;")
        gl.addRow("飞行高度(m):", self.lbl_gt_alt)
        btn_local = QPushButton("📍 生成局部占据网格图")
        btn_local.setMinimumHeight(32)
        btn_local.setStyleSheet(
            "QPushButton{background:#FF9800;color:white;border-radius:4px;font-weight:bold}"
            "QPushButton:hover{background:#F57C00}")
        btn_local.clicked.connect(self.build_local_ogm)
        gl.addRow(btn_local)
        lay.addWidget(gg)

        tg_ctrl = QGroupBox("✈️ 轨迹显示控制")
        tl_ctrl = QVBoxLayout(tg_ctrl)
        self.chk_traj_all = QCheckBox("全局轨迹显示总开关")
        self.chk_traj_all.setChecked(True)
        self.chk_traj_all.setStyleSheet("font-weight: bold;")
        tl_ctrl.addWidget(self.chk_traj_all)
        self.chk_traj_actual = QCheckBox("实际飞行轨迹 (蓝色实线)")
        self.chk_traj_actual.setChecked(True)
        tl_ctrl.addWidget(self.chk_traj_actual)
        self.chk_traj_expected = QCheckBox("期望轨迹 (红色虚线)")
        self.chk_traj_expected.setChecked(True)
        tl_ctrl.addWidget(self.chk_traj_expected)
        self.chk_traj_all.toggled.connect(self._on_traj_all_toggled)
        self.chk_traj_actual.toggled.connect(self._on_traj_toggle_changed)
        self.chk_traj_expected.toggled.connect(self._on_traj_toggle_changed)
        self._load_traj_settings()
        lay.addWidget(tg_ctrl)

        btn_map_clr = QPushButton("🗑️ 清除建图数据")
        btn_map_clr.setMinimumHeight(35)
        btn_map_clr.setStyleSheet(
            "QPushButton{background:#e74c3c;color:white;border-radius:4px;font-weight:bold}"
            "QPushButton:hover{background:#c0392b}")
        btn_map_clr.clicked.connect(self.clear_mapping_data)
        lay.addWidget(btn_map_clr)

        lay.addStretch()
        scroll.setWidget(content)
        return scroll

    # ==================== Right Panel ====================

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.tab_widget = QTabWidget()

        it = QWidget(); il = QVBoxLayout(it)
        self.text_info = QTextEdit(); self.text_info.setReadOnly(True)
        self.text_info.setFont(QFont("Consolas", 9))
        il.addWidget(self.text_info)
        self.tab_widget.addTab(it, "📝 信息")

        self.plot_tab = QWidget()
        self.plot_tab_lay = QVBoxLayout(self.plot_tab)
        self.lbl_plot_ph = QLabel("请先加载数据并选择话题\n然后点击绘图按钮生成图表")
        self.lbl_plot_ph.setAlignment(Qt.AlignCenter)
        self.lbl_plot_ph.setStyleSheet("QLabel{font-size:14px;color:#666;padding:20px;background:#f5f5f5;border-radius:8px}")
        self.plot_tab_lay.addWidget(self.lbl_plot_ph)
        self.tab_widget.addTab(self.plot_tab, "📈 位姿数据")

        self.tab_3d = QWidget()
        self.tab_3d_lay = QVBoxLayout(self.tab_3d)
        self.lbl_3d_ph = QLabel("点击\"🌐 绘制3D轨迹\"按钮\n生成交互式3D轨迹图")
        self.lbl_3d_ph.setAlignment(Qt.AlignCenter)
        self.lbl_3d_ph.setStyleSheet("QLabel{font-size:14px;color:#666;padding:20px;background:#f5f5f5;border-radius:8px}")
        self.tab_3d_lay.addWidget(self.lbl_3d_ph)
        self.tab_widget.addTab(self.tab_3d, "🌐 3D轨迹")

        self.tab_oct = QWidget()
        self.tab_oct_lay = QVBoxLayout(self.tab_oct)
        self.lbl_oct_ph = QLabel("在\"建图处理\"中配置参数\n点击\"🏗️ 生成八叉树地图\"")
        self.lbl_oct_ph.setAlignment(Qt.AlignCenter)
        self.lbl_oct_ph.setStyleSheet("QLabel{font-size:14px;color:#666;padding:20px;background:#f5f5f5;border-radius:8px}")
        self.tab_oct_lay.addWidget(self.lbl_oct_ph)
        self.tab_widget.addTab(self.tab_oct, "🗺️ 八叉树")

        self.tab_ogm = QWidget()
        self.tab_ogm_lay = QVBoxLayout(self.tab_ogm)
        self.lbl_ogm_ph = QLabel("在\"建图处理\"中配置参数\n点击\"🌍 生成全局/局部占据网格图\"")
        self.lbl_ogm_ph.setAlignment(Qt.AlignCenter)
        self.lbl_ogm_ph.setStyleSheet("QLabel{font-size:14px;color:#666;padding:20px;background:#f5f5f5;border-radius:8px}")
        self.tab_ogm_lay.addWidget(self.lbl_ogm_ph)
        self.tab_widget.addTab(self.tab_ogm, "🧱 占据网格图")

        self.tab_filter = QWidget()
        self.tab_filter_lay = QVBoxLayout(self.tab_filter)
        fsplit = QSplitter(Qt.Vertical)
        self.text_filter = QTextEdit(); self.text_filter.setReadOnly(True)
        self.text_filter.setFont(QFont("Consolas", 9)); self.text_filter.setMaximumHeight(250)
        fsplit.addWidget(self.text_filter)
        self.filter_chart_cont = QWidget()
        self.filter_chart_lay = QVBoxLayout(self.filter_chart_cont)
        self.filter_chart_lay.setContentsMargins(0, 0, 0, 0)
        self.lbl_filt_ph = QLabel("应用滤波后，对比图表将在此显示")
        self.lbl_filt_ph.setAlignment(Qt.AlignCenter)
        self.lbl_filt_ph.setStyleSheet("QLabel{font-size:13px;color:#666;padding:15px;background:#f5f5f5;border-radius:8px}")
        self.filter_chart_lay.addWidget(self.lbl_filt_ph)
        fsplit.addWidget(self.filter_chart_cont)
        self.tab_filter_lay.addWidget(fsplit)
        self.tab_widget.addTab(self.tab_filter, "🔍 滤波对比")

        layout.addWidget(self.tab_widget)
        return panel

    # ==================== Console ====================

    def _setup_console_redirect(self):
        self._signaler = ConsoleSignaler()
        self._signaler.text_written.connect(self._append_console)
        self._out_stream = ConsoleStream(self._signaler)
        self._err_stream = ConsoleStream(self._signaler)
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        sys.stdout = self._out_stream
        sys.stderr = self._err_stream

    def _append_console(self, text):
        self.text_console.moveCursor(QTextCursor.End)
        self.text_console.insertPlainText(text)
        self.text_console.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        sys.stdout = self._orig_out
        sys.stderr = self._orig_err
        super().closeEvent(event)

    # ==================== Menu & Status ====================

    def _create_menu_bar(self):
        mb = self.menuBar()
        fm = mb.addMenu("文件(&F)")
        for label, short, slot in [
            ("打开Bag文件", "Ctrl+O", self.open_bag_file),
            ("导出数据", "Ctrl+E", self.export_data),
            ("保存图表", "Ctrl+S", self.save_plots),
        ]:
            a = QAction(label + "...", self); a.setShortcut(short); a.triggered.connect(slot); fm.addAction(a)
        fm.addSeparator()
        a = QAction("退出", self); a.setShortcut("Ctrl+Q"); a.triggered.connect(self.close); fm.addAction(a)

        vm = mb.addMenu("视图(&V)")
        for label, slot in [
            ("位置曲线", self.plot_position), ("姿态曲线", self.plot_orientation),
            ("2D轨迹", self.plot_2d), ("3D轨迹", self.plot_3d),
            ("仪表板", self.plot_dashboard), ("清除曲线", self.clear_all_curves),
        ]:
            a = QAction(label, self); a.triggered.connect(slot); vm.addAction(a)

        hm = mb.addMenu("帮助(&H)")
        a = QAction("关于", self); a.triggered.connect(self.show_about); hm.addAction(a)

    def _create_status_bar(self):
        self.statusBar().showMessage("就绪 - 请打开一个.bag文件开始分析")
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    # ==================== Dynamic Topic Buttons ====================

    def _rebuild_topic_buttons(self):
        for btn in self._topic_buttons_pose:
            self.pose_btn_layout.removeWidget(btn); btn.deleteLater()
        self._topic_buttons_pose.clear()
        for btn in self._topic_buttons_map:
            self.map_btn_layout.removeWidget(btn); btn.deleteLater()
        self._topic_buttons_map.clear()

        if not self.parser:
            for layout, text in [
                (self.pose_btn_layout, "请先加载bag文件"),
                (self.map_btn_layout, "请先加载bag文件"),
            ]:
                lbl = QLabel(text); lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("color:#999;font-style:italic;")
                layout.addWidget(lbl)
            return

        for name, info in sorted(self.parser.topics_info.items()):
            is_pose = info.msg_type in POSE_MSG_TYPES
            is_map = info.msg_type in MAPPING_MSG_TYPES

            if is_pose:
                btn = QPushButton(f"{name}\n  ({info.msg_type})")
                btn.setStyleSheet(
                    "QPushButton{text-align:left;padding:4px 8px;border:1px solid #ccc;border-radius:3px}"
                    "QPushButton:hover{background:#e3f2fd;border-color:#2196F3}")
                btn.clicked.connect(lambda _, t=name: self.select_topic(t))
                self.pose_btn_layout.addWidget(btn)
                self._topic_buttons_pose.append(btn)

            if is_map:
                btn = QPushButton(f"{name}\n  ({info.msg_type})")
                btn.setStyleSheet(
                    "QPushButton{text-align:left;padding:4px 8px;border:1px solid #4CAF50;border-radius:3px;color:#2E7D32}"
                    "QPushButton:hover{background:#E8F5E9;border-color:#4CAF50}")
                btn.clicked.connect(lambda _, t=name: self.select_map_topic(t))
                self.map_btn_layout.addWidget(btn)
                self._topic_buttons_map.append(btn)

            if not is_pose and not is_map:
                btn = QPushButton(f"{name}\n  ({info.msg_type})")
                btn.setStyleSheet(
                    "QPushButton{text-align:left;padding:4px 8px;border:1px solid #ddd;border-radius:3px;color:#999;background:#fafafa}"
                    "QPushButton:hover{background:#fff3e0;border-color:#ff9800}")
                btn.clicked.connect(lambda _, t=name, mt=info.msg_type: self._unsupported(t, mt))
                self.pose_btn_layout.addWidget(btn)
                self._topic_buttons_pose.append(btn)

    def _unsupported(self, name, msg_type):
        supported = "\n".join(f"  - {t}" for t in sorted(POSE_MSG_TYPES | MAPPING_MSG_TYPES))
        QMessageBox.warning(self, "不支持的消息类型",
            f"话题 '{name}' 的消息类型为 '{msg_type}'，\n当前不支持提取。\n\n支持的消息类型：\n{supported}", QMessageBox.Ok)

    def _on_pipe_topic_changed(self):
        if not self.parser:
            return
        upper_topic = self.combo_pipe_upper.currentData()
        lower_topic = self.combo_pipe_lower.currentData()
        self._pipe_upper_data = None
        self._pipe_lower_data = None
        if upper_topic and upper_topic in self.parser.get_topic_names():
            try:
                pd = self.parser.extract_pose_data(upper_topic)
                if pd and len(pd.x) > 0:
                    self._pipe_upper_data = pd
            except Exception: pass
        if lower_topic and lower_topic in self.parser.get_topic_names():
            try:
                pd = self.parser.extract_pose_data(lower_topic)
                if pd and len(pd.x) > 0:
                    self._pipe_lower_data = pd
            except Exception: pass
        parts = []
        if self._pipe_upper_data is not None:
            parts.append(f"上界: {len(self._pipe_upper_data.x)}点")
        if self._pipe_lower_data is not None:
            parts.append(f"下界: {len(self._pipe_lower_data.x)}点")
        if parts:
            self.lbl_pipe_status.setText(" | ".join(parts))
            self.lbl_pipe_status.setStyleSheet("color: #2E7D32; font-size: 9px;")
        else:
            self.lbl_pipe_status.setText("未选择管道话题")
            self.lbl_pipe_status.setStyleSheet("color: #999; font-size: 9px;")

    def _populate_pipe_combos(self):
        if not self.parser:
            return
        for combo in [self.combo_pipe_upper, self.combo_pipe_lower]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("未选择", None)
            for name in sorted(self.parser.get_topic_names()):
                info = self.parser.topics_info.get(name)
                if info and info.msg_type in POSE_MSG_TYPES:
                    combo.addItem(f"{name} ({info.msg_type})", name)
            combo.blockSignals(False)
        self._on_pipe_topic_changed()

    def select_map_topic(self, topic_name):
        if not self.parser:
            return
        info = self.parser.topics_info.get(topic_name)
        if info and 'PointCloud2' in info.msg_type:
            try:
                pc2 = self.parser.extract_pointcloud2_data(topic_name)
                if pc2 is not None and len(pc2) > 0:
                    self._pc2_points = pc2
                    self._pc2_topic = topic_name
                    self.lbl_gt_pos.setText(f"点云已就绪: {len(pc2)} 点")
                    self.statusBar().showMessage(f"已从 '{topic_name}' 提取 {len(pc2)} 个点云数据")
                    QMessageBox.information(self, "点云提取成功",
                        f"从话题 '{topic_name}' 提取了 {len(pc2)} 个点。", QMessageBox.Ok)
                else:
                    QMessageBox.warning(self, "提取失败", f"无法从 '{topic_name}' 提取点云数据。", QMessageBox.Ok)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"提取点云时出错:\n{str(e)}", QMessageBox.Ok)

    # ==================== Data Clearing ====================

    def _clear_canvas(self, tab_lay, canvas_attr, toolbar_attr, placeholder):
        canvas = getattr(self, canvas_attr, None)
        toolbar = getattr(self, toolbar_attr, None)
        if canvas is not None:
            tab_lay.removeWidget(canvas); canvas.deleteLater()
            setattr(self, canvas_attr, None)
        if toolbar is not None:
            tab_lay.removeWidget(toolbar); toolbar.deleteLater()
            setattr(self, toolbar_attr, None)
        if placeholder.parent() is None:
            tab_lay.addWidget(placeholder)
        placeholder.show()

    def clear_all_curves(self):
        self.visualizer.clear_all_data()
        self.current_pose_data.clear()
        self.filtered_pose_data.clear()
        self.filter_processor.filter_history.clear()
        self.filter_processor.last_config = None
        self._clear_canvas(self.plot_tab_lay, 'plot_canvas', '_plot_toolbar', self.lbl_plot_ph)
        self._clear_canvas(self.tab_3d_lay, 'canvas_3d', '_toolbar_3d', self.lbl_3d_ph)
        self._clear_canvas(self.tab_oct_lay, 'canvas_octree', '_toolbar_octree', self.lbl_oct_ph)
        self._clear_canvas(self.tab_ogm_lay, 'canvas_ogm', '_toolbar_ogm', self.lbl_ogm_ph)
        self.text_filter.clear()
        self.statusBar().showMessage("所有曲线和数据已清除")
        QMessageBox.information(self, "清除完成", "所有位姿数据和图表已清除。", QMessageBox.Ok)

    def clear_mapping_data(self):
        self.octree_map = None
        self.octree_visualizer = None
        self.global_ogm = None
        self.local_ogm = None
        self._drone_altitude = None
        self._actual_traj_xy = None
        self._expected_traj_xy = None
        self._clear_canvas(self.tab_oct_lay, 'canvas_octree', '_toolbar_octree', self.lbl_oct_ph)
        self._clear_canvas(self.tab_ogm_lay, 'canvas_ogm', '_toolbar_ogm', self.lbl_ogm_ph)
        self.lbl_gt_pos.setText("未获取")
        self.lbl_gt_alt.setText("未获取")
        if hasattr(self, '_pc2_points'):
            del self._pc2_points
        if hasattr(self, '_pc2_topic'):
            del self._pc2_topic
        self.statusBar().showMessage("建图数据已清除")
        QMessageBox.information(self, "清除完成", "所有建图数据和可视化已清除。", QMessageBox.Ok)

    # ==================== File Operations ====================

    def open_bag_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "选择ROS Bag文件", "", "Bag文件 (*.bag);;所有文件 (*)")
        if fp:
            self.load_bag(fp)

    def load_bag(self, path):
        self.current_pose_data.clear()
        self.filtered_pose_data.clear()
        self.visualizer.clear_all_data()
        self._clear_canvas(self.plot_tab_lay, 'plot_canvas', '_plot_toolbar', self.lbl_plot_ph)
        self._clear_canvas(self.tab_3d_lay, 'canvas_3d', '_toolbar_3d', self.lbl_3d_ph)
        self._clear_canvas(self.tab_oct_lay, 'canvas_octree', '_toolbar_octree', self.lbl_oct_ph)
        self._clear_canvas(self.tab_ogm_lay, 'canvas_ogm', '_toolbar_ogm', self.lbl_ogm_ph)
        self.text_filter.clear()
        self.text_info.clear()
        self.octree_map = None; self.octree_visualizer = None
        self.global_ogm = None; self.local_ogm = None; self._drone_altitude = None
        self._actual_traj_xy = None; self._expected_traj_xy = None
        self.lbl_file.setText(f"正在加载:\n{path}")
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self._thread = BagLoadingThread(path)
        self._thread.loading_finished.connect(self._on_loaded)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.start()

    def _on_loaded(self, parser):
        self.parser = parser
        self.progress_bar.setVisible(False)
        self.lbl_file.setText(f"已加载:\n{parser.bag_file_path}")
        self.statusBar().showMessage(f"加载完成 - {len(parser.topics_info)} 个话题")
        self.list_topics.clear()
        for name, info in sorted(parser.topics_info.items()):
            self.list_topics.addItem(f"{name}\n  ({info.msg_type}, {info.message_count}条)")
            self.list_topics.item(self.list_topics.count()-1).setData(Qt.UserRole, name)
        self._rebuild_topic_buttons()
        self._populate_pipe_combos()
        self.text_info.setText(parser.get_statistics_report())
        self._update_time_range()
        QMessageBox.information(self, "加载成功",
            f"发现 {len(parser.topics_info)} 个话题\n旧数据已清除。", QMessageBox.Ok)

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.lbl_file.setText("加载失败")
        QMessageBox.critical(self, "加载错误", msg, QMessageBox.Ok)

    # ==================== Topic Selection ====================

    def select_topic(self, name):
        if not self.parser:
            return
        try:
            pd = self.parser.extract_pose_data(name)
            if pd and len(pd.timestamp) > 0:
                self.current_pose_data[name] = pd
                self.visualizer.add_pose_data(name, pd, 'raw')
                if name in self.filtered_pose_data:
                    del self.filtered_pose_data[name]
                info = self.parser.topics_info[name]
                is_twist = 'Twist' in info.msg_type
                if is_twist:
                    txt = (f"话题: {name}\n{'='*40}\n\n消息类型: {info.msg_type} (速度数据)\n"
                           f"数据点数: {len(pd.timestamp)}\n"
                           f"线速度 Vx: [{pd.x.min():.4f}, {pd.x.max():.4f}] m/s\n"
                           f"角速度 Wz: [{pd.yaw.min():.4f}, {pd.yaw.max():.4f}] rad/s")
                else:
                    txt = (f"话题: {name}\n{'='*40}\n\n数据点数: {len(pd.timestamp)}\n"
                           f"X: [{pd.x.min():.4f}, {pd.x.max():.4f}] m\n"
                           f"Yaw: [{np.degrees(pd.yaw).min():.2f}, {np.degrees(pd.yaw).max():.2f}] deg")
                self.text_info.setText(txt)
                self.statusBar().showMessage(f"已提取 '{name}' ({len(pd.timestamp)} 点)")
                QMessageBox.information(self, "提取成功",
                    f"从 '{name}' 提取了 {len(pd.timestamp)} 个数据点。", QMessageBox.Ok)
            else:
                QMessageBox.warning(self, "提取失败", f"无法从 '{name}' 提取数据。", QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e), QMessageBox.Ok)

    # ==================== Filter ====================

    def _get_filter_config(self) -> FilterConfig:
        tmap = {0:'moving_average',1:'weighted_moving_average',2:'exponential_moving_average',
                3:'median',4:'kalman',5:'lowpass_butterworth',6:'savitzky_golay'}
        return FilterConfig(
            filter_type=tmap[self.combo_filter.currentIndex()],
            window_size=self.spin_win.value(), alpha=self.dspin_alpha.value(),
            cutoff_freq=self.dspin_cutoff.value(),
            process_noise=self.dspin_q.value(), measurement_noise=self.dspin_r.value())

    def apply_filter(self):
        if not self.current_pose_data:
            QMessageBox.warning(self, "提示", "请先提取位姿数据！", QMessageBox.Ok); return
        config = self._get_filter_config()
        name = self.combo_filter.currentText()
        try:
            for tn, pd in self.current_pose_data.items():
                fp = self.filter_processor.apply_filter_to_pose_data(pd, config)
                self.filtered_pose_data[tn] = fp
                self.visualizer.add_pose_data(tn, fp, 'filtered')
            self.text_filter.setText(f"滤波器: {name}\n已应用到所有话题。")
            self._show_filter_chart()
            self.tab_widget.setCurrentIndex(5)
            QMessageBox.information(self, "滤波完成", f"已应用 {name} 滤波器。", QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, "滤波错误", str(e), QMessageBox.Ok)

    def _show_filter_chart(self):
        if not self.current_pose_data or not self.filtered_pose_data:
            return
        if self.filter_canvas is not None:
            self.filter_chart_lay.removeWidget(self.filter_canvas); self.filter_canvas.deleteLater()
        if hasattr(self, '_filter_tb') and self._filter_tb is not None:
            self.filter_chart_lay.removeWidget(self._filter_tb); self._filter_tb.deleteLater()
        self.lbl_filt_ph.hide()
        topics = list(self.filtered_pose_data.keys())
        n = len(topics)
        fig = Figure(figsize=(12, 4*n), dpi=100)
        for ti, tn in enumerate(topics):
            r = self.current_pose_data.get(tn); f = self.filtered_pose_data.get(tn)
            if r is None or f is None: continue
            for fi, fld in enumerate(['x','y','z']):
                ax = fig.add_subplot(n, 3, ti*3+fi+1)
                rv = getattr(r, fld, np.array([])); fv = getattr(f, fld, np.array([]))
                if len(rv) > 0 and len(fv) > 0:
                    t = r.timestamp - r.timestamp[0]
                    ml = min(len(rv), len(fv), len(t))
                    ax.plot(t[:ml], rv[:ml], 'b-', label='原始', lw=1, alpha=0.7)
                    ax.plot(t[:ml], fv[:ml], 'r--', label='滤波', lw=1, alpha=0.7)
                    ax.set_title(f'{tn} - {fld.upper()}', fontsize=9)
                    ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=7)
        fig.tight_layout()
        self.filter_canvas = FigureCanvas(fig); self.filter_canvas.setMinimumHeight(300)
        self._filter_tb = NavigationToolbar(self.filter_canvas, self); self._filter_tb.setMaximumHeight(35)
        self.filter_chart_lay.addWidget(self._filter_tb); self.filter_chart_lay.addWidget(self.filter_canvas)

    # ==================== Plot Operations ====================

    def _no_data(self):
        QMessageBox.warning(self, "无数据", "请先加载bag文件并提取数据！", QMessageBox.Ok)

    def _show_fig(self, fig, title):
        self._clear_canvas(self.plot_tab_lay, 'plot_canvas', '_plot_toolbar', self.lbl_plot_ph)
        self.lbl_plot_ph.hide()
        c = FigureCanvas(fig); c.setMinimumHeight(400)
        tb = NavigationToolbar(c, self); tb.setMaximumHeight(35)
        self.plot_tab_lay.insertWidget(0, tb); self.plot_tab_lay.insertWidget(1, c)
        self.plot_canvas = c; self._plot_toolbar = tb
        self.tab_widget.setCurrentIndex(1)
        self.statusBar().showMessage(f"图表已生成: {title}")

    def _get_display_mode(self):
        idx = self.combo_display_mode.currentIndex()
        return {0: 'filtered', 1: 'raw', 2: 'both'}.get(idx, 'both')

    def plot_position(self):
        if not self.current_pose_data: self._no_data(); return
        mode = self._get_display_mode()
        try:
            show_f = mode in ('filtered', 'both')
            show_r = mode in ('raw', 'both')
            fig = self.visualizer.plot_position_time(show_filtered=show_f)
            if not show_r and mode == 'filtered':
                for ax in fig.get_axes():
                    lines = [l for l in ax.get_lines() if l.get_linestyle() == '-']
                    for l in lines:
                        l.set_visible(False)
            self._show_fig(fig, "位置曲线")
        except Exception as e: QMessageBox.critical(self, "错误", str(e), QMessageBox.Ok)

    def plot_orientation(self):
        if not self.current_pose_data: self._no_data(); return
        mode = self._get_display_mode()
        try:
            show_f = mode in ('filtered', 'both')
            fig = self.visualizer.plot_orientation_time(show_filtered=show_f)
            if mode == 'filtered':
                for ax in fig.get_axes():
                    for l in ax.get_lines():
                        if l.get_linestyle() == '-':
                            l.set_visible(False)
            self._show_fig(fig, "姿态曲线")
        except Exception as e: QMessageBox.critical(self, "错误", str(e), QMessageBox.Ok)

    def plot_2d(self):
        if not self.current_pose_data: self._no_data(); return
        mode = self._get_display_mode()
        try:
            show_f = mode in ('filtered', 'both')
            self._show_fig(self.visualizer.plot_2d_trajectory(show_filtered=show_f), "2D轨迹")
        except Exception as e: QMessageBox.critical(self, "错误", str(e), QMessageBox.Ok)

    def plot_3d(self):
        if not self.current_pose_data: self._no_data(); return
        try:
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            self._clear_canvas(self.tab_3d_lay, 'canvas_3d', '_toolbar_3d', self.lbl_3d_ph)
            self.lbl_3d_ph.hide()
            fig = Figure(figsize=(10,8), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            colors = ['b','g','r','c','m','y','k','orange','purple']
            mode = self._get_display_mode()
            for i, (tn, pd) in enumerate(self.current_pose_data.items()):
                c = colors[i%len(colors)]
                if len(pd.x) > 0:
                    if mode in ('raw', 'both'):
                        ax.plot(pd.x, pd.y, pd.z, color=c, ls='-', label=f'{tn}', lw=1.5, alpha=0.8)
                    if mode in ('filtered', 'both') and tn in self.filtered_pose_data:
                        fp = self.filtered_pose_data[tn]
                        if len(fp.x) > 0:
                            ax.plot(fp.x, fp.y, fp.z, color=c, ls='--', label=f'{tn}(滤波)', lw=1.5, alpha=0.8)
                    ax.scatter([pd.x[0]],[pd.y[0]],[pd.z[0]],color=c,marker='o',s=100,zorder=5,edgecolors='k')
                    ax.scatter([pd.x[-1]],[pd.y[-1]],[pd.z[-1]],color=c,marker='s',s=100,zorder=5,edgecolors='k')
            
            if self.chk_pipe_show.isChecked() and (self._pipe_upper_data is not None or self._pipe_lower_data is not None):
                self._draw_3d_pipe(ax)
            
            ax.set_xlabel('X(m)'); ax.set_ylabel('Y(m)'); ax.set_zlabel('Z(m)')
            ax.legend(fontsize=9); ax.view_init(elev=20, azim=45); fig.tight_layout()
            cv = FigureCanvas(fig); cv.setMinimumHeight(400)
            tb = NavigationToolbar(cv, self); tb.setMaximumHeight(35)
            self.tab_3d_lay.insertWidget(0, tb); self.tab_3d_lay.insertWidget(1, cv)
            self.canvas_3d = cv; self._toolbar_3d = tb
            self.tab_widget.setCurrentIndex(2)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "错误", str(e), QMessageBox.Ok)

    def _draw_3d_pipe(self, ax):
        """在3D轨迹视图中绘制管道范围"""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        alpha = self.dspin_pipe_alpha.value()
        lw = self.spin_pipe_lw.value()
        
        upper = self._pipe_upper_data
        lower = self._pipe_lower_data
        
        if upper is not None and lower is not None and len(upper.x) > 1 and len(lower.x) > 1:
            n = min(len(upper.x), len(lower.x))
            ux, uy, uz = upper.x[:n], upper.y[:n], upper.z[:n]
            lx, ly, lz = lower.x[:n], lower.y[:n], lower.z[:n]
            
            step = max(1, n // 100)
            for i in range(0, n - step, step):
                j = min(i + step, n - 1)
                verts = [
                    [ux[i], uy[i], uz[i]],
                    [ux[j], uy[j], uz[j]],
                    [lx[j], ly[j], lz[j]],
                    [lx[i], ly[i], lz[i]],
                ]
                poly = Poly3DCollection([verts], alpha=alpha * 0.3,
                                        facecolor='cyan', edgecolor='teal',
                                        linewidth=lw * 0.5)
                ax.add_collection3d(poly)
            
            ax.plot(ux, uy, uz, color='teal', ls='-.', lw=lw, alpha=alpha, label='管道上界')
            ax.plot(lx, ly, lz, color='purple', ls='-.', lw=lw, alpha=alpha, label='管道下界')
            
        elif upper is not None and len(upper.x) > 1:
            ax.plot(upper.x, upper.y, upper.z, color='teal', ls='-.', lw=lw,
                   alpha=alpha, label='管道上界')
        elif lower is not None and len(lower.x) > 1:
            ax.plot(lower.x, lower.y, lower.z, color='purple', ls='-.', lw=lw,
                   alpha=alpha, label='管道下界')

    def plot_dashboard(self):
        if not self.current_pose_data: self._no_data(); return
        try: self._show_fig(self.visualizer.create_dashboard(), "仪表板")
        except Exception as e: QMessageBox.critical(self, "错误", str(e), QMessageBox.Ok)

    # ==================== Octree ====================

    def _get_octree_config(self):
        sm = {0:'height',1:'density',2:'depth',3:'custom'}
        qm = {0:'low',1:'medium',2:'high'}
        hmin = self.spin_hmin.value(); hmax = self.spin_hmax.value()
        if hmin <= -998: hmin = float('-inf')
        if hmax >= 998: hmax = float('inf')
        return OctreeConfig(
            voxel_size=self.spin_voxel.value(), max_depth=self.spin_maxd.value(),
            min_depth=self.spin_mind.value(), density_threshold=self.spin_dthresh.value(),
            height_min=hmin, height_max=hmax,
            color_scheme=sm.get(self.combo_cscheme.currentIndex(),'height'),
            transparency=self.dspin_transp.value(),
            show_borders=self.chk_borders.isChecked(),
            render_quality=qm.get(self.combo_rqual.currentIndex(),'medium'))

    def _get_point_cloud(self) -> Optional[np.ndarray]:
        if hasattr(self, '_pc2_points') and self._pc2_points is not None and len(self._pc2_points) > 0:
            return self._pc2_points
        if self.parser is not None:
            for tn, info in self.parser.topics_info.items():
                if 'PointCloud2' in info.msg_type:
                    try:
                        pc2 = self.parser.extract_pointcloud2_data(tn)
                        if pc2 is not None and len(pc2) > 0:
                            self._pc2_points = pc2; self._pc2_topic = tn
                            return pc2
                    except Exception: pass
        if not self.current_pose_data: return None
        pts = []
        for tn, pd in self.current_pose_data.items():
            if len(pd.x) > 0: pts.append(np.column_stack([pd.x, pd.y, pd.z]))
        return np.vstack(pts) if pts else None

    def build_octree(self):
        pts = self._get_point_cloud()
        if pts is None or len(pts) == 0:
            QMessageBox.warning(self, "无数据",
                "请先加载bag文件并提取数据！\n\n支持: PointCloud2话题 / 位姿话题", QMessageBox.Ok); return
        config = self._get_octree_config()
        try:
            self.octree_map = OctreeMap(config); self.octree_map.build(pts)
            self.octree_visualizer = OctreeVisualizer(self.octree_map)
            self._clear_canvas(self.tab_oct_lay, 'canvas_octree', '_toolbar_octree', self.lbl_oct_ph)
            self.lbl_oct_ph.hide()
            fig = self.octree_visualizer.visualize()
            cv = FigureCanvas(fig); cv.setMinimumHeight(400)
            tb = NavigationToolbar(cv, self); tb.setMaximumHeight(35)
            self.tab_oct_lay.insertWidget(0, tb); self.tab_oct_lay.insertWidget(1, cv)
            self.canvas_octree = cv; self._toolbar_octree = tb
            self.tab_widget.setCurrentIndex(3)
            stats = self.octree_map.get_statistics()
            QMessageBox.information(self, "构建完成",
                f"叶节点: {stats.get('total_leaves',0)}\n耗时: {stats.get('build_time',0):.3f}s", QMessageBox.Ok)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "构建错误", str(e), QMessageBox.Ok)

    def octree_2d(self):
        if not self.octree_visualizer: QMessageBox.warning(self, "提示", "请先生成八叉树地图！", QMessageBox.Ok); return
        try: self._show_fig(self.octree_visualizer.visualize_2d_projection(), "八叉树2D投影")
        except Exception as e: QMessageBox.critical(self, "错误", str(e), QMessageBox.Ok)

    def octree_stats(self):
        if not self.octree_visualizer: QMessageBox.warning(self, "提示", "请先生成八叉树地图！", QMessageBox.Ok); return
        try: self._show_fig(self.octree_visualizer.visualize_statistics(), "八叉树统计")
        except Exception as e: QMessageBox.critical(self, "错误", str(e), QMessageBox.Ok)

    # ==================== Occupancy Grid ====================

    def _get_ogm_config(self):
        hmin = self.spin_ogm_hmin.value(); hmax = self.spin_ogm_hmax.value()
        if hmin <= -998: hmin = float('-inf')
        if hmax >= 998: hmax = float('inf')
        return OccupancyGridConfig(
            resolution=self.spin_ogm_res.value(),
            height_min=hmin, height_max=hmax,
            occupancy_threshold=self.spin_ogm_occ.value(),
            local_range=self.spin_local_range.value())

    def _ensure_gt_data(self) -> bool:
        """确保ground_truth数据已加载"""
        gt_topic = '/robot1/robot/ground_truth'
        if gt_topic in self.current_pose_data:
            return True
        if self.parser and gt_topic in self.parser.get_topic_names():
            try:
                pd = self.parser.extract_pose_data(gt_topic)
                if pd and len(pd.x) > 0:
                    self.current_pose_data[gt_topic] = pd
                    return True
            except Exception: pass
        return False

    def _get_gt_position(self) -> Optional[np.ndarray]:
        """获取当前时间点对应的无人机位置（地图坐标系）"""
        gt_topic = '/robot1/robot/ground_truth'
        if not self._ensure_gt_data():
            return None
        pd = self.current_pose_data.get(gt_topic)
        if pd is None or len(pd.x) == 0:
            return None
        t_point = self.spin_time_point.value()
        idx = np.searchsorted(pd.timestamp, t_point, side='right') - 1
        idx = max(0, min(idx, len(pd.x) - 1))
        self._drone_altitude = pd.z[idx]
        self.lbl_gt_alt.setText(f"{self._drone_altitude:.2f}")
        raw = np.array([pd.x[idx], pd.y[idx]])
        return self.coord_transformer.transform_point(raw)

    def _update_time_range(self):
        """更新时间范围和滑块"""
        gt_topic = '/robot1/robot/ground_truth'
        if not self._ensure_gt_data():
            self.lbl_time_range.setText("未加载")
            return
        pd = self.current_pose_data.get(gt_topic)
        if pd is None or len(pd.timestamp) == 0:
            self.lbl_time_range.setText("无时间数据")
            return
        t_min = float(pd.timestamp[0])
        t_max = float(pd.timestamp[-1])
        self._gt_time_range = (t_min, t_max)
        self._gt_timestamps = pd.timestamp
        duration = t_max - t_min
        self.lbl_time_range.setText(f"{t_min:.2f}s ~ {t_max:.2f}s (共{duration:.2f}s)")
        self.spin_time_point.setRange(t_min, t_max)
        self.spin_time_point.setValue(t_min)
        self.slider_time.blockSignals(True)
        self.slider_time.setRange(0, 1000)
        self.slider_time.setValue(0)
        self.slider_time.blockSignals(False)

    def _on_time_slider_changed(self, value):
        if self._gt_time_range[1] <= self._gt_time_range[0]:
            return
        t_min, t_max = self._gt_time_range
        t_point = t_min + (t_max - t_min) * (value / 1000.0)
        self.spin_time_point.blockSignals(True)
        self.spin_time_point.setValue(t_point)
        self.spin_time_point.blockSignals(False)
        self._update_position_display(t_point)

    def _on_time_spin_changed(self, value):
        if self._gt_time_range[1] <= self._gt_time_range[0]:
            return
        t_min, t_max = self._gt_time_range
        ratio = (value - t_min) / max(t_max - t_min, 1e-6)
        self.slider_time.blockSignals(True)
        self.slider_time.setValue(int(np.clip(ratio * 1000, 0, 1000)))
        self.slider_time.blockSignals(False)
        self._update_position_display(value)

    def _update_position_display(self, t_point):
        """根据时间点更新无人机位置和高度显示"""
        gt_topic = '/robot1/robot/ground_truth'
        pd = self.current_pose_data.get(gt_topic)
        if pd is None or len(pd.x) == 0:
            return
        idx = np.searchsorted(pd.timestamp, t_point, side='right') - 1
        idx = max(0, min(idx, len(pd.x) - 1))
        self._drone_altitude = pd.z[idx]
        self.lbl_gt_alt.setText(f"{self._drone_altitude:.2f}")
        raw = np.array([pd.x[idx], pd.y[idx]])
        pos = self.coord_transformer.transform_point(raw)
        self.lbl_gt_pos.setText(f"({pos[0]:.2f}, {pos[1]:.2f})")

    def _extract_trajectory_data(self):
        """提取实际和期望轨迹数据并应用坐标转换
        
        实际轨迹：根据当前时间点截取（从起始到当前时间）
        期望轨迹：完整显示所有数据
        """
        gt_topic = '/robot1/robot/ground_truth'
        exp_topic = '/position_exp'
        t_point = self.spin_time_point.value()
        
        if gt_topic in self.current_pose_data:
            pd = self.current_pose_data[gt_topic]
            if len(pd.x) > 0:
                mask = pd.timestamp <= t_point
                if np.any(mask):
                    raw = np.column_stack([pd.x[mask], pd.y[mask]])
                    self._actual_traj_xy = self.coord_transformer.transform_trajectory(raw)
                    print(f"[OK] 实际轨迹: {np.sum(mask)}/{len(pd.x)} 点 (t<={t_point:.2f}s)")
                else:
                    self._actual_traj_xy = None
                    print(f"[WARN] 时间点 {t_point:.2f}s 之前无实际轨迹数据")
        
        if self.parser is not None:
            topic_names = self.parser.get_topic_names()
            exp_candidates = [t for t in topic_names if 'position_exp' in t or 'expected' in t or 'reference' in t]
            if exp_topic not in topic_names and exp_candidates:
                exp_topic = exp_candidates[0]
                print(f"[INFO] 使用话题 '{exp_topic}' 作为期望轨迹数据源")
            
            if exp_topic in topic_names and exp_topic not in self.current_pose_data:
                try:
                    pd = self.parser.extract_pose_data(exp_topic)
                    if pd and len(pd.x) > 0:
                        self.current_pose_data[exp_topic] = pd
                        print(f"[OK] 期望轨迹数据已提取: {len(pd.x)} 点")
                    else:
                        print(f"[WARN] 话题 '{exp_topic}' 提取结果为空")
                except Exception as e:
                    print(f"[WARN] 提取期望轨迹失败: {str(e)}")
        
        if exp_topic in self.current_pose_data:
            pd = self.current_pose_data[exp_topic]
            if len(pd.x) > 0:
                raw = np.column_stack([pd.x, pd.y])
                self._expected_traj_xy = self.coord_transformer.transform_trajectory(raw)
                print(f"[OK] 期望轨迹已转换: {len(raw)} 点, 范围 X=[{self._expected_traj_xy[:,0].min():.1f},{self._expected_traj_xy[:,0].max():.1f}] Y=[{self._expected_traj_xy[:,1].min():.1f},{self._expected_traj_xy[:,1].max():.1f}]")
        else:
            self._expected_traj_xy = None
            if self.parser is not None:
                print(f"[WARN] 未找到期望轨迹话题 (已搜索: /position_exp, *position_exp*, *expected*, *reference*)")
    
    def _build_trajectory_overlay(self) -> TrajectoryOverlay:
        """构建轨迹叠加配置"""
        self._extract_trajectory_data()
        return TrajectoryOverlay(
            actual_xy=self._actual_traj_xy,
            expected_xy=self._expected_traj_xy,
            show_actual=self.chk_traj_actual.isChecked() and self.chk_traj_all.isChecked(),
            show_expected=self.chk_traj_expected.isChecked() and self.chk_traj_all.isChecked(),
            show_all=self.chk_traj_all.isChecked(),
            actual_color=(0, 0, 1.0),
            expected_color=(1.0, 0, 0),
            actual_linewidth=2.0,
            expected_linewidth=2.0,
            arrow_interval=5.0,
            arrow_length=6.0,
        )
    
    def _on_traj_all_toggled(self, checked):
        self.chk_traj_actual.setEnabled(checked)
        self.chk_traj_expected.setEnabled(checked)
        self._on_traj_toggle_changed()
    
    def _on_traj_toggle_changed(self):
        self._save_traj_settings()
        if (self.global_ogm is not None and self.global_ogm.grid is not None) or \
           (self.local_ogm is not None and self.local_ogm.grid is not None):
            self._refresh_ogm_display()
    
    def _refresh_ogm_display(self):
        """刷新占据网格图显示（应用轨迹设置变更）"""
        overlay = self._build_trajectory_overlay()
        if self.global_ogm is not None and self.global_ogm.grid is not None and \
           self.local_ogm is not None and self.local_ogm.grid is not None:
            self._show_ogm_dual(overlay)
        elif self.global_ogm is not None and self.global_ogm.grid is not None:
            center = self._get_gt_position()
            self._show_ogm_single(self.global_ogm, "全局占据网格图",
                                  center=center, overlay=overlay)
        elif self.local_ogm is not None and self.local_ogm.grid is not None:
            center = self._local_center if hasattr(self, '_local_center') and self._local_center is not None else None
            self._show_ogm_single(self.local_ogm, "局部占据网格图",
                                  center=center, overlay=overlay, is_local=True)
    
    def _save_traj_settings(self):
        settings = QSettings("ROSBagAnalyzer", "TrajectorySettings")
        settings.setValue("show_all", self.chk_traj_all.isChecked())
        settings.setValue("show_actual", self.chk_traj_actual.isChecked())
        settings.setValue("show_expected", self.chk_traj_expected.isChecked())
    
    def _load_traj_settings(self):
        settings = QSettings("ROSBagAnalyzer", "TrajectorySettings")
        self.chk_traj_all.setChecked(settings.value("show_all", True, type=bool))
        self.chk_traj_actual.setChecked(settings.value("show_actual", True, type=bool))
        self.chk_traj_expected.setChecked(settings.value("show_expected", True, type=bool))

    def build_global_ogm(self):
        pts = self._get_point_cloud()
        if pts is None or len(pts) == 0:
            QMessageBox.warning(self, "无数据", "请先提取点云数据！", QMessageBox.Ok); return
        config = self._get_ogm_config()
        sensor_pos = None
        center = self._get_gt_position()
        if center is not None:
            sensor_pos = center.reshape(1, -1)
        try:
            self.global_ogm = OccupancyGridMap(config)
            self.global_ogm.build(pts, sensor_positions=sensor_pos)
            overlay = self._build_trajectory_overlay()
            if self.local_ogm is not None and self.local_ogm.grid is not None:
                self._show_ogm_dual(overlay)
            else:
                self._show_ogm_single(self.global_ogm, "全局占据网格图",
                                      center=center, overlay=overlay)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "错误", str(e), QMessageBox.Ok)

    def build_local_ogm(self):
        center = self._get_gt_position()
        if center is None:
            QMessageBox.warning(self, "无位置数据",
                "无法获取无人机位置！\n请先提取 /robot1/robot/ground_truth 话题数据。",
                QMessageBox.Ok); return
        self.lbl_gt_pos.setText(f"({center[0]:.2f}, {center[1]:.2f})")
        config = self._get_ogm_config()
        try:
            self.local_ogm = OccupancyGridMap(config)
            if self.global_ogm is not None and self.global_ogm.grid is not None:
                self.local_ogm.extract_local_from_global(
                    self.global_ogm, center, local_range=config.local_range)
            else:
                pts = self._get_point_cloud()
                if pts is None or len(pts) == 0:
                    QMessageBox.warning(self, "无数据",
                        "请先生成全局占据网格图，或提取点云数据！", QMessageBox.Ok); return
                self.local_ogm.build_local(pts, center, sensor_position=center)
            overlay = self._build_trajectory_overlay()
            if self.global_ogm is not None and self.global_ogm.grid is not None:
                self._show_ogm_dual(overlay)
            else:
                self._show_ogm_single(self.local_ogm,
                    f"局部占据网格图 (中心: {center[0]:.1f}, {center[1]:.1f})",
                    center=center, overlay=overlay, is_local=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "错误", str(e), QMessageBox.Ok)

    def _show_ogm_single(self, ogm, title, center=None, overlay=None, is_local=False):
        if ogm is None or ogm.grid is None:
            return
        self._clear_canvas(self.tab_ogm_lay, 'canvas_ogm', '_toolbar_ogm', self.lbl_ogm_ph)
        self.lbl_ogm_ph.hide()
        local_bounds = None
        if ogm is self.global_ogm and self.local_ogm is not None:
            local_bounds = self.local_ogm.local_bounds
        fig = ogm.visualize(title=title, local_bounds=local_bounds, center=center,
                           trajectory_overlay=overlay, is_local=is_local)
        cv = FigureCanvas(fig); cv.setMinimumHeight(400)
        tb = NavigationToolbar(cv, self); tb.setMaximumHeight(35)
        self.tab_ogm_lay.insertWidget(0, tb); self.tab_ogm_lay.insertWidget(1, cv)
        self.canvas_ogm = cv; self._toolbar_ogm = tb
        self.tab_widget.setCurrentIndex(4)
        stats = ogm.get_statistics()
        self.statusBar().showMessage(
            f"占据网格图已生成 | 占据: {stats.get('occupancy_rate',0):.1f}% | 空闲: {stats.get('free_rate',0):.1f}%")

    def _show_ogm_dual(self, overlay=None):
        if self.global_ogm is None or self.local_ogm is None:
            return
        if self.global_ogm.grid is None or self.local_ogm.grid is None:
            return
        self._clear_canvas(self.tab_ogm_lay, 'canvas_ogm', '_toolbar_ogm', self.lbl_ogm_ph)
        self.lbl_ogm_ph.hide()
        fig = self.local_ogm.visualize_dual(self.global_ogm, trajectory_overlay=overlay)
        cv = FigureCanvas(fig); cv.setMinimumHeight(400)
        tb = NavigationToolbar(cv, self); tb.setMaximumHeight(35)
        self.tab_ogm_lay.insertWidget(0, tb); self.tab_ogm_lay.insertWidget(1, cv)
        self.canvas_ogm = cv; self._toolbar_ogm = tb
        self.tab_widget.setCurrentIndex(4)
        g_stats = self.global_ogm.get_statistics()
        l_stats = self.local_ogm.get_statistics()
        self.statusBar().showMessage(
            f"全局+局部占据网格图 | 全局占据: {g_stats.get('occupancy_rate',0):.1f}% | 局部占据: {l_stats.get('occupancy_rate',0):.1f}%")

    # ==================== Export ====================

    def export_data(self):
        if not self.parser: QMessageBox.warning(self, "提示", "请先加载bag文件！", QMessageBox.Ok); return
        d = QFileDialog.getExistingDirectory(self, "选择导出目录", "./exported_data")
        if d:
            if self.parser.export_topics_to_csv(d):
                QMessageBox.information(self, "导出成功", f"数据已导出至:\n{d}", QMessageBox.Ok)

    def save_plots(self):
        if not self.visualizer.figures:
            QMessageBox.information(self, "提示", "暂无图表可保存。", QMessageBox.Ok); return
        d = QFileDialog.getExistingDirectory(self, "选择保存目录", "./plots")
        if d:
            n = self.visualizer.save_all_plots(d)
            QMessageBox.information(self, "保存成功", f"已保存 {n} 个图表", QMessageBox.Ok)

    def show_about(self):
        QMessageBox.about(self, "关于",
            "<h2>ROS Bag数据分析与可视化系统</h2>"
            "<p>版本: 3.4.0</p>"
            "<ul>"
            "<li>位姿数据可视化（位置/姿态/2D/3D轨迹）</li>"
            "<li>八叉树地图生成与可视化</li>"
            "<li>全局/局部占据网格图</li>"
            "<li>PointCloud2点云数据支持</li>"
            "<li>7种滤波算法</li>"
            "</ul>"
            "<p>Python + PyQt5 + Matplotlib</p>")


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("ROS Bag Analyzer")
    app.setApplicationVersion("3.4.0")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
