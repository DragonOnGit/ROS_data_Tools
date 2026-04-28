# -*- coding: utf-8 -*-
"""
ROS Bag文件数据处理与可视化系统 - 主程序
功能：提供图形用户界面，整合所有功能模块
作者：Auto-generated
日期：2026-04-27

使用方法：
    python main.py
    
或通过命令行指定bag文件：
    python main.py -f <bag_file_path>
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QListWidget, QTextEdit, QFileDialog, QComboBox,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
                             QMessageBox, QSplitter, QProgressBar, QStatusBar,
                             QAction, QMenu, QMenuBar, QDialog, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

from bag_parser import BagParser, TopicInfo, PoseData
from data_visualizer import DataVisualizer
from filter_processor import FilterProcessor, FilterConfig

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Configure Chinese font support
_font_candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                    'Noto Sans CJK SC', 'PingFang SC', 'Heiti SC',
                    'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = _font_candidates
plt.rcParams['axes.unicode_minus'] = False


class BagLoadingThread(QThread):
    """Bag文件加载线程"""
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
            topics_info = parser.parse_bag()
            self.loading_finished.emit(parser)
        except Exception as e:
            self.error_occurred.emit(f"加载失败: {str(e)}")


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        self.parser = None
        self.visualizer = DataVisualizer()
        self.filter_processor = FilterProcessor()
        self.current_pose_data = {}
        self.filtered_pose_data = {}
        
        self.plot_canvas = None
        self._plot_toolbar = None
        self.filter_canvas = None
        self._filter_toolbar = None
        
        self.setWindowTitle("ROS Bag数据分析与可视化系统")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 700)
        
        self._setup_ui()
        self._create_menu_bar()
        self._create_status_bar()
    
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1050])
        
        main_layout.addWidget(splitter)
    
    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # === 文件操作组 ===
        file_group = QGroupBox("📁 文件操作")
        file_layout = QVBoxLayout(file_group)
        
        btn_open = QPushButton("📂 打开Bag文件")
        btn_open.setMinimumHeight(35)
        btn_open.clicked.connect(self.open_bag_file)
        
        self.lbl_current_file = QLabel("未选择文件")
        self.lbl_current_file.setWordWrap(True)
        
        file_layout.addWidget(btn_open)
        file_layout.addWidget(self.lbl_current_file)
        layout.addWidget(file_group)
        
        # === 话题列表组 ===
        topic_group = QGroupBox("📋 话题列表")
        topic_layout = QVBoxLayout(topic_group)
        
        self.list_topics = QListWidget()
        self.list_topics.setSelectionMode(QListWidget.MultiSelection)
        self.list_topics.itemSelectionChanged.connect(self.on_topic_selection_changed)
        
        topic_layout.addWidget(self.list_topics)
        layout.addWidget(topic_group)
        
        # === 目标话题快速选择 ===
        target_group = QGroupBox("🎯 常用目标话题")
        target_layout = QVBoxLayout(target_group)
        
        self.target_topic_buttons = []
        target_topics = ['/Odometry', '/path', '/path_exp', 
                        '/robot1/robot/cmd_vel', '/robot1/robot/ground_truth']
        
        for topic in target_topics:
            btn = QPushButton(topic)
            btn.setToolTip(f"选择并提取 {topic} 的位姿数据")
            btn.clicked.connect(lambda checked, t=topic: self.select_target_topic(t))
            target_layout.addWidget(btn)
            self.target_topic_buttons.append(btn)
        
        layout.addWidget(target_group)
        
        # === 滤波设置组 ===
        filter_group = QGroupBox("🔧 滤波设置")
        filter_layout = QFormLayout(filter_group)
        
        self.combo_filter_type = QComboBox()
        self.combo_filter_type.addItems([
            '滑动平均 (Moving Average)',
            '加权滑动平均 (Weighted MA)',
            '指数平滑 (Exponential MA)',
            '中值滤波 (Median)',
            '卡尔曼滤波 (Kalman)',
            '巴特沃斯低通 (Butterworth)',
            'Savitzky-Golay'
        ])
        filter_layout.addRow("滤波类型:", self.combo_filter_type)
        
        self.spin_window_size = QSpinBox()
        self.spin_window_size.setRange(1, 51)
        self.spin_window_size.setValue(7)
        filter_layout.addRow("窗口大小:", self.spin_window_size)
        
        self.double_alpha = QDoubleSpinBox()
        self.double_alpha.setRange(0.01, 1.0)
        self.double_alpha.setValue(0.3)
        self.double_alpha.setSingleStep(0.05)
        filter_layout.addRow("平滑系数 α:", self.double_alpha)
        
        self.double_cutoff_freq = QDoubleSpinBox()
        self.double_cutoff_freq.setRange(0.1, 50.0)
        self.double_cutoff_freq.setValue(5.0)
        filter_layout.addRow("截止频率 (Hz):", self.double_cutoff_freq)
        
        self.double_process_noise = QDoubleSpinBox()
        self.double_process_noise.setRange(0.001, 1.0)
        self.double_process_noise.setValue(0.01)
        self.double_process_noise.setDecimals(4)
        self.double_process_noise.setSingleStep(0.005)
        filter_layout.addRow("过程噪声 Q:", self.double_process_noise)
        
        self.double_meas_noise = QDoubleSpinBox()
        self.double_meas_noise.setRange(0.001, 10.0)
        self.double_meas_noise.setValue(0.1)
        self.double_meas_noise.setDecimals(3)
        filter_layout.addRow("测量噪声 R:", self.double_meas_noise)
        
        btn_apply_filter = QPushButton("✨ 应用滤波")
        btn_apply_filter.setMinimumHeight(35)
        btn_apply_filter.clicked.connect(self.apply_filter_to_selected)
        filter_layout.addRow(btn_apply_filter)
        
        layout.addWidget(filter_group)
        
        # === 操作按钮组 ===
        action_group = QGroupBox("⚡ 快速操作")
        action_layout = QVBoxLayout(action_group)
        
        btn_plot_position = QPushButton("📈 绘制位置曲线")
        btn_plot_position.clicked.connect(self.plot_position_data)
        
        btn_plot_orientation = QPushButton("📊 绘制姿态曲线")
        btn_plot_orientation.clicked.connect(self.plot_orientation_data)
        
        btn_plot_trajectory = QPushButton("🗺️ 绘制2D轨迹")
        btn_plot_trajectory.clicked.connect(self.plot_2d_trajectory)
        
        btn_plot_dashboard = QPushButton("🎛️ 综合仪表板")
        btn_plot_dashboard.clicked.connect(self.plot_dashboard)
        
        btn_clear_curves = QPushButton("🗑️ 清除曲线")
        btn_clear_curves.setMinimumHeight(30)
        btn_clear_curves.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        btn_clear_curves.clicked.connect(self.clear_all_curves)
        
        btn_export_data = QPushButton("💾 导出数据(CSV)")
        btn_export_data.clicked.connect(self.export_data)
        
        btn_save_plots = QPushButton("🖼️ 保存所有图表")
        btn_save_plots.clicked.connect(self.save_all_plots)
        
        action_layout.addWidget(btn_plot_position)
        action_layout.addWidget(btn_plot_orientation)
        action_layout.addWidget(btn_plot_trajectory)
        action_layout.addWidget(btn_plot_dashboard)
        action_layout.addWidget(btn_clear_curves)
        action_layout.addWidget(btn_export_data)
        action_layout.addWidget(btn_save_plots)
        
        layout.addWidget(action_group)
        layout.addStretch()
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        self.tab_widget = QTabWidget()
        
        # === 信息显示标签页 ===
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        
        self.text_info = QTextEdit()
        self.text_info.setReadOnly(True)
        self.text_info.setFont(QFont("Consolas", 9))
        info_layout.addWidget(self.text_info)
        
        self.tab_widget.addTab(info_tab, "📝 信息")
        
        # === 图表显示标签页 ===
        self.plot_tab = QWidget()
        self.plot_tab_layout = QVBoxLayout(self.plot_tab)
        
        self.lbl_plot_placeholder = QLabel("请先加载数据并选择话题\n然后点击绘图按钮生成图表")
        self.lbl_plot_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_plot_placeholder.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666666;
                padding: 20px;
                background-color: #f5f5f5;
                border-radius: 8px;
            }
        """)
        self.plot_tab_layout.addWidget(self.lbl_plot_placeholder)
        
        self.tab_widget.addTab(self.plot_tab, "📊 图表")
        
        # === 滤波对比标签页 ===
        self.compare_tab = QWidget()
        self.compare_tab_layout = QVBoxLayout(self.compare_tab)
        
        compare_splitter = QSplitter(Qt.Vertical)
        
        self.text_filter_report = QTextEdit()
        self.text_filter_report.setReadOnly(True)
        self.text_filter_report.setFont(QFont("Consolas", 9))
        self.text_filter_report.setMaximumHeight(250)
        compare_splitter.addWidget(self.text_filter_report)
        
        self.filter_chart_container = QWidget()
        self.filter_chart_layout = QVBoxLayout(self.filter_chart_container)
        self.filter_chart_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_filter_placeholder = QLabel("应用滤波后，对比图表将在此显示")
        self.lbl_filter_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_filter_placeholder.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #666666;
                padding: 15px;
                background-color: #f5f5f5;
                border-radius: 8px;
            }
        """)
        self.filter_chart_layout.addWidget(self.lbl_filter_placeholder)
        
        compare_splitter.addWidget(self.filter_chart_container)
        
        self.compare_tab_layout.addWidget(compare_splitter)
        
        self.tab_widget.addTab(self.compare_tab, "🔍 滤波对比")
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    def _create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("文件(&F)")
        
        open_action = QAction("打开Bag文件...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_bag_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("导出数据...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        save_plots_action = QAction("保存图表...", self)
        save_plots_action.setShortcut("Ctrl+S")
        save_plots_action.triggered.connect(self.save_all_plots)
        file_menu.addAction(save_plots_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu("视图(&V)")
        
        pos_view_action = QAction("位置-时间曲线", self)
        pos_view_action.triggered.connect(self.plot_position_data)
        view_menu.addAction(pos_view_action)
        
        ori_view_action = QAction("姿态-时间曲线", self)
        ori_view_action.triggered.connect(self.plot_orientation_data)
        view_menu.addAction(ori_view_action)
        
        traj_view_action = QAction("2D轨迹图", self)
        traj_view_action.triggered.connect(self.plot_2d_trajectory)
        view_menu.addAction(traj_view_action)
        
        dashboard_action = QAction("综合仪表板", self)
        dashboard_action.triggered.connect(self.plot_dashboard)
        view_menu.addAction(dashboard_action)
        
        view_menu.addSeparator()
        clear_action = QAction("清除所有曲线", self)
        clear_action.triggered.connect(self.clear_all_curves)
        view_menu.addAction(clear_action)
        
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于...", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def _create_status_bar(self):
        self.statusBar().showMessage("就绪 - 请打开一个.bag文件开始分析")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    # ==================== Data Clearing ====================
    
    def _clear_all_data(self):
        """清除所有已加载的数据、缓存和图表
        
        在重新加载bag文件时调用，确保工作区完全重置。
        """
        # 清除位姿数据
        self.current_pose_data.clear()
        self.filtered_pose_data.clear()
        
        # 清除可视化器数据
        self.visualizer.clear_all_data()
        
        # 清除滤波处理器历史
        self.filter_processor.filter_history.clear()
        self.filter_processor.last_config = None
        
        # 清除图表canvas
        self._clear_plot_canvas()
        self._clear_filter_canvas()
        
        # 清除文本报告
        self.text_filter_report.clear()
        
        # 重置信息显示
        self.text_info.clear()
        
        print("✅ 所有数据已清除，工作区已重置")
    
    def _clear_plot_canvas(self):
        """清除图表选项卡中的canvas"""
        if self.plot_canvas is not None:
            self.plot_tab_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.deleteLater()
            self.plot_canvas = None
        if self._plot_toolbar is not None:
            self.plot_tab_layout.removeWidget(self._plot_toolbar)
            self._plot_toolbar.deleteLater()
            self._plot_toolbar = None
        
        # 恢复placeholder
        if self.lbl_plot_placeholder.parent() is None:
            self.plot_tab_layout.addWidget(self.lbl_plot_placeholder)
        self.lbl_plot_placeholder.show()
        self.lbl_plot_placeholder.setText("请先加载数据并选择话题\n然后点击绘图按钮生成图表")
    
    def _clear_filter_canvas(self):
        """清除滤波对比选项卡中的canvas"""
        if self.filter_canvas is not None:
            self.filter_chart_layout.removeWidget(self.filter_canvas)
            self.filter_canvas.deleteLater()
            self.filter_canvas = None
        if self._filter_toolbar is not None:
            self.filter_chart_layout.removeWidget(self._filter_toolbar)
            self._filter_toolbar.deleteLater()
            self._filter_toolbar = None
        
        if self.lbl_filter_placeholder.parent() is None:
            self.filter_chart_layout.addWidget(self.lbl_filter_placeholder)
        self.lbl_filter_placeholder.show()
        self.lbl_filter_placeholder.setText("应用滤波后，对比图表将在此显示")
    
    def clear_all_curves(self):
        """清除当前图窗中所有已绘制的曲线"""
        # 清除可视化器中存储的所有数据和图表
        self.visualizer.clear_all_data()
        
        # 清除当前位姿数据
        self.current_pose_data.clear()
        self.filtered_pose_data.clear()
        
        # 清除滤波历史
        self.filter_processor.filter_history.clear()
        self.filter_processor.last_config = None
        
        # 清除图表canvas
        self._clear_plot_canvas()
        self._clear_filter_canvas()
        
        # 清除文本报告
        self.text_filter_report.clear()
        
        self.statusBar().showMessage("✅ 所有曲线和数据已清除")
        self.tab_widget.setCurrentIndex(0)
        
        QMessageBox.information(
            self,
            "清除完成",
            "所有曲线数据已彻底清除！\n\n"
            "包括：已加载的位姿数据、滤波结果、图表曲线。\n"
            "您可以重新选择话题并绘制新图表。",
            QMessageBox.Ok
        )
    
    # ==================== File Operations ====================
    
    def open_bag_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择ROS Bag文件",
            "",
            "Bag文件 (*.bag);;所有文件 (*)"
        )
        
        if file_path:
            self.load_bag_file(file_path)
    
    def load_bag_file(self, file_path: str):
        # 重新加载时先清除所有旧数据
        self._clear_all_data()
        
        self.lbl_current_file.setText(f"正在加载:\n{file_path}")
        self.statusBar().showMessage(f"正在加载: {os.path.basename(file_path)}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.loading_thread = BagLoadingThread(file_path)
        self.loading_thread.loading_finished.connect(self.on_bag_loaded)
        self.loading_thread.error_occurred.connect(self.on_load_error)
        self.loading_thread.progress_update.connect(
            lambda msg: self.statusBar().showMessage(msg)
        )
        self.loading_thread.start()
    
    def on_bag_loaded(self, parser: BagParser):
        self.parser = parser
        self.progress_bar.setVisible(False)
        
        self.lbl_current_file.setText(f"已加载:\n{parser.bag_file_path}")
        self.statusBar().showMessage(
            f"加载完成 - 发现 {len(parser.topics_info)} 个话题（旧数据已清除）"
        )
        
        # 更新话题列表
        self.list_topics.clear()
        for topic_name, info in sorted(parser.topics_info.items()):
            item_text = f"{info.name}\n  ({info.msg_type}, {info.message_count}条)"
            self.list_topics.addItem(item_text)
            item = self.list_topics.item(self.list_topics.count() - 1)
            item.setData(Qt.UserRole, topic_name)
        
        report = parser.get_statistics_report()
        self.text_info.setText(report)
        
        self.tab_widget.setCurrentIndex(0)
        
        QMessageBox.information(
            self,
            "加载成功",
            f"成功加载bag文件！\n\n"
            f"发现 {len(parser.topics_info)} 个话题\n\n"
            f"提示：之前的数据已全部清除，可放心使用新数据。",
            QMessageBox.Ok
        )
    
    def on_load_error(self, error_msg: str):
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("加载失败")
        self.lbl_current_file.setText("加载失败")
        
        QMessageBox.critical(self, "加载错误", error_msg, QMessageBox.Ok)
    
    # ==================== Topic Selection ====================
    
    def on_topic_selection_changed(self):
        selected_items = self.list_topics.selectedItems()
        if selected_items:
            self.statusBar().showMessage(f"已选择 {len(selected_items)} 个话题")
        else:
            self.statusBar().showMessage("未选择任何话题")
    
    def select_target_topic(self, topic_name: str):
        if not self.parser:
            QMessageBox.warning(self, "提示", "请先加载bag文件！", QMessageBox.Ok)
            return
        
        if topic_name not in self.parser.get_topic_names():
            available = ", ".join(self.parser.get_topic_names()[:5])
            QMessageBox.warning(
                self,
                "话题不存在",
                f"话题 '{topic_name}' 未在bag文件中找到。\n\n可用的话题包括:\n{available}...",
                QMessageBox.Ok
            )
            return
        
        try:
            pose_data = self.parser.extract_pose_data(topic_name)
            
            if pose_data and len(pose_data.timestamp) > 0:
                self.current_pose_data[topic_name] = pose_data
                self.visualizer.add_pose_data(topic_name, pose_data, 'raw')
                
                if topic_name in self.filtered_pose_data:
                    del self.filtered_pose_data[topic_name]
                
                info_text = (
                    f"话题: {topic_name}\n"
                    f"{'='*40}\n\n"
                    f"数据点数: {len(pose_data.timestamp)}\n"
                    f"时间范围: {pose_data.timestamp[0]:.2f}s - {pose_data.timestamp[-1]:.2f}s\n"
                    f"持续时间: {(pose_data.timestamp[-1]-pose_data.timestamp[0]):.2f}s\n\n"
                    f"位置范围:\n"
                    f"  X: [{pose_data.x.min():.4f}, {pose_data.x.max():.4f}] m\n"
                    f"  Y: [{pose_data.y.min():.4f}, {pose_data.y.max():.4f}] m\n"
                    f"  Z: [{pose_data.z.min():.4f}, {pose_data.z.max():.4f}] m\n\n"
                    f"姿态范围:\n"
                    f"  Roll:  [{np.degrees(pose_data.roll).min():.2f}°, {np.degrees(pose_data.roll).max():.2f}°]\n"
                    f"  Pitch: [{np.degrees(pose_data.pitch).min():.2f}°, {np.degrees(pose_data.pitch).max():.2f}°]\n"
                    f"  Yaw:   [{np.degrees(pose_data.yaw).min():.2f}°, {np.degrees(pose_data.yaw).max():.2f}°]"
                )
                self.text_info.setText(info_text)
                
                self.statusBar().showMessage(
                    f"已提取 '{topic_name}' 的位姿数据 ({len(pose_data.timestamp)} 个点)"
                )
                
                for i in range(self.list_topics.count()):
                    item = self.list_topics.item(i)
                    if item.data(Qt.UserRole) == topic_name:
                        item.setSelected(True)
                        self.list_topics.setCurrentItem(item)
                        break
                
                QMessageBox.information(
                    self,
                    "数据提取成功",
                    f"成功从话题 '{topic_name}' 提取了 {len(pose_data.timestamp)} 个数据点的位姿信息。",
                    QMessageBox.Ok
                )
            else:
                QMessageBox.warning(
                    self,
                    "提取失败",
                    f"无法从话题 '{topic_name}' 提取有效的位姿数据。\n\n"
                    "可能原因:\n"
                    "- 该话题不包含位姿信息\n"
                    "- 消息格式不支持\n"
                    "- 数据为空",
                    QMessageBox.Ok
                )
                
        except Exception as e:
            QMessageBox.critical(
                self, "错误", f"提取数据时发生错误:\n{str(e)}", QMessageBox.Ok
            )
    
    # ==================== Filter Operations ====================
    
    def get_current_filter_config(self) -> FilterConfig:
        type_map = {
            0: 'moving_average',
            1: 'weighted_moving_average',
            2: 'exponential_moving_average',
            3: 'median',
            4: 'kalman',
            5: 'lowpass_butterworth',
            6: 'savitzky_golay'
        }
        
        config = FilterConfig(
            filter_type=type_map[self.combo_filter_type.currentIndex()],
            window_size=self.spin_window_size.value(),
            alpha=self.double_alpha.value(),
            cutoff_freq=self.double_cutoff_freq.value(),
            process_noise=self.double_process_noise.value(),
            measurement_noise=self.double_meas_noise.value()
        )
        
        return config
    
    def apply_filter_to_selected(self):
        if not self.current_pose_data:
            QMessageBox.warning(
                self, "提示", "请先选择并提取至少一个话题的位姿数据！", QMessageBox.Ok
            )
            return
        
        config = self.get_current_filter_config()
        filter_name = self.combo_filter_type.currentText().split('(')[0].strip()
        
        report_lines = [f"滤波器: {filter_name}", "=" * 60, ""]
        
        try:
            for topic_name, pose_data in self.current_pose_data.items():
                print(f"\n正在对 '{topic_name}' 应用 {filter_name} 滤波...")
                
                filtered_pose = self.filter_processor.apply_filter_to_pose_data(
                    pose_data, config
                )
                
                self.filtered_pose_data[topic_name] = filtered_pose
                self.visualizer.add_pose_data(topic_name, filtered_pose, 'filtered')
                
                fields = ['x', 'y', 'z']
                report_lines.append(f"\n话题: {topic_name}")
                report_lines.append("-" * 40)
                
                for field in fields:
                    original = getattr(pose_data, field)
                    filtered = getattr(filtered_pose, field)
                    
                    if len(original) > 0 and len(filtered) > 0:
                        stats = self.filter_processor.get_filter_comparison_stats(
                            original, filtered
                        )
                        
                        report_lines.append(
                            f"\n  字段: {field.upper()}"
                            f"\n    标准差降低: {stats.get('std_reduction_pct', 0):.2f}%"
                            f"\n    RMSE: {stats.get('rmse', 0):.6f}"
                            f"\n    相关性: {stats.get('correlation', 0):.4f}"
                        )
            
            full_report = "\n".join(report_lines)
            self.text_filter_report.setText(full_report)
            
            self._display_filter_comparison_chart()
            
            self.tab_widget.setCurrentIndex(2)
            self.statusBar().showMessage(f"已完成滤波处理 - {filter_name}")
            
            QMessageBox.information(
                self,
                "滤波完成",
                f"已对所有选中的话题应用 {filter_name} 滤波器。\n\n"
                f"请在'滤波对比'标签页查看详细效果和对比图表。",
                QMessageBox.Ok
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "滤波错误", f"应用滤波器时发生错误:\n{str(e)}", QMessageBox.Ok
            )
    
    def _display_filter_comparison_chart(self):
        """在滤波对比选项卡中显示对比图表"""
        if not self.current_pose_data or not self.filtered_pose_data:
            return
        
        # 清除旧的canvas
        if self.filter_canvas is not None:
            self.filter_chart_layout.removeWidget(self.filter_canvas)
            self.filter_canvas.deleteLater()
            self.filter_canvas = None
        if self._filter_toolbar is not None:
            self.filter_chart_layout.removeWidget(self._filter_toolbar)
            self._filter_toolbar.deleteLater()
            self._filter_toolbar = None
        
        self.lbl_filter_placeholder.hide()
        
        topics = list(self.filtered_pose_data.keys())
        n_topics = len(topics)
        if n_topics == 0:
            return
        
        fig = Figure(figsize=(12, 4 * n_topics), dpi=100)
        
        # Apply Chinese font to the figure
        fig.set_facecolor('white')
        
        for t_idx, topic_name in enumerate(topics):
            raw = self.current_pose_data.get(topic_name)
            filt = self.filtered_pose_data.get(topic_name)
            
            if raw is None or filt is None:
                continue
            
            fields = ['x', 'y', 'z']
            for f_idx, field in enumerate(fields):
                ax_idx = t_idx * 3 + f_idx + 1
                ax = fig.add_subplot(n_topics, 3, ax_idx)
                
                raw_values = getattr(raw, field, np.array([]))
                filt_values = getattr(filt, field, np.array([]))
                
                if len(raw_values) > 0 and len(filt_values) > 0:
                    time_raw = raw.timestamp - raw.timestamp[0]
                    min_len = min(len(raw_values), len(filt_values), len(time_raw))
                    
                    ax.plot(time_raw[:min_len], raw_values[:min_len], 
                           'b-', label='原始', linewidth=1.0, alpha=0.7)
                    ax.plot(time_raw[:min_len], filt_values[:min_len], 
                           'r--', label='滤波', linewidth=1.0, alpha=0.7)
                    ax.set_title(f'{topic_name} - {field.upper()}', fontsize=9)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.legend(fontsize=7, loc='upper right')
                    ax.set_xlabel('时间 (s)', fontsize=8)
                    
                    if f_idx == 0:
                        ax.set_ylabel('坐标 (m)', fontsize=8)
        
        fig.tight_layout()
        
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(300)
        
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setMaximumHeight(35)
        
        self.filter_chart_layout.addWidget(toolbar)
        self.filter_chart_layout.addWidget(canvas)
        
        self.filter_canvas = canvas
        self._filter_toolbar = toolbar
    
    # ==================== Plot Operations ====================
    
    def plot_position_data(self):
        if not self.current_pose_data:
            self._show_no_data_warning()
            return
        try:
            fig = self.visualizer.plot_position_time(show_filtered=True)
            self._display_figure(fig, "位置数据曲线")
        except Exception as e:
            self._show_error_dialog("绘制图表错误", str(e))
    
    def plot_orientation_data(self):
        if not self.current_pose_data:
            self._show_no_data_warning()
            return
        try:
            fig = self.visualizer.plot_orientation_time(show_filtered=True)
            self._display_figure(fig, "姿态数据曲线")
        except Exception as e:
            self._show_error_dialog("绘制图表错误", str(e))
    
    def plot_2d_trajectory(self):
        if not self.current_pose_data:
            self._show_no_data_warning()
            return
        try:
            fig = self.visualizer.plot_2d_trajectory(show_filtered=True)
            self._display_figure(fig, "2D运动轨迹")
        except Exception as e:
            self._show_error_dialog("绘制图表错误", str(e))
    
    def plot_dashboard(self):
        if not self.current_pose_data:
            self._show_no_data_warning()
            return
        try:
            fig = self.visualizer.create_dashboard()
            self._display_figure(fig, "综合仪表板")
        except Exception as e:
            self._show_error_dialog("绘制图表错误", str(e))
    
    def _display_figure(self, fig, title: str):
        """在界面上显示matplotlib图表，嵌入到Qt界面的图表选项卡中"""
        # 清除旧的canvas
        if self.plot_canvas is not None:
            self.plot_tab_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.deleteLater()
            self.plot_canvas = None
        if self._plot_toolbar is not None:
            self.plot_tab_layout.removeWidget(self._plot_toolbar)
            self._plot_toolbar.deleteLater()
            self._plot_toolbar = None
        
        # 隐藏placeholder
        self.lbl_plot_placeholder.hide()
        
        # 创建新的canvas
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(400)
        
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setMaximumHeight(35)
        
        self.plot_tab_layout.insertWidget(0, toolbar)
        self.plot_tab_layout.insertWidget(1, canvas)
        
        self.plot_canvas = canvas
        self._plot_toolbar = toolbar
        
        self.tab_widget.setCurrentIndex(1)
        self.statusBar().showMessage(f"图表已生成: {title}")
        
        # 保存临时文件
        temp_dir = './temp_plots'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        safe_title = title.replace(' ', '_').replace('/', '_')
        temp_path = os.path.join(temp_dir, f'{safe_title}.png')
        fig.savefig(temp_path, dpi=150, bbox_inches='tight')
    
    # ==================== Export Operations ====================
    
    def export_data(self):
        if not self.parser:
            QMessageBox.warning(self, "提示", "请先加载bag文件！", QMessageBox.Ok)
            return
        
        output_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", "./exported_data"
        )
        
        if output_dir:
            success = self.parser.export_topics_to_csv(output_dir)
            if success:
                QMessageBox.information(
                    self, "导出成功", f"数据已成功导出至:\n{output_dir}", QMessageBox.Ok
                )
            else:
                QMessageBox.critical(
                    self, "导出失败", "导出过程中出现错误，请检查日志。", QMessageBox.Ok
                )
    
    def save_all_plots(self):
        if not self.visualizer.figures:
            QMessageBox.information(
                self, "提示", "暂无图表可保存。请先生成一些图表。", QMessageBox.Ok
            )
            return
        
        output_dir = QFileDialog.getExistingDirectory(
            self, "选择保存目录", "./plots"
        )
        
        if output_dir:
            saved_files = self.visualizer.save_all_plots(output_dir)
            QMessageBox.information(
                self, "保存成功", f"已保存 {len(saved_files)} 个图表至:\n{output_dir}",
                QMessageBox.Ok
            )
    
    # ==================== Utility ====================
    
    def _show_no_data_warning(self):
        QMessageBox.warning(
            self, "无数据", "请先加载bag文件并选择要分析的话题！", QMessageBox.Ok
        )
    
    def _show_error_dialog(self, title: str, message: str):
        QMessageBox.critical(self, title, message, QMessageBox.Ok)
    
    def show_about_dialog(self):
        QMessageBox.about(
            self,
            "关于",
            "<h2>ROS Bag数据分析与可视化系统</h2>"
            "<p>版本: 1.1.0</p>"
            "<p>功能:</p>"
            "<ul>"
            "<li>.bag文件解析与话题提取</li>"
            "<li>位姿数据可视化（位置、姿态、轨迹）</li>"
            "<li>多种滤波算法支持</li>"
            "<li>数据导出与报表生成</li>"
            "<li>图表内嵌预览与交互</li>"
            "</ul>"
            "<p>基于Python + PyQt5 + Matplotlib开发</p>"
        )


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("ROS Bag Analyzer")
    app.setApplicationVersion("1.1.0")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
