# -*- coding: utf-8 -*-
import sys
import os
import math
import time
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.cm as cm

from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QCheckBox, QFileDialog,
    QProgressBar, QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QSpinBox, QDoubleSpinBox, QMessageBox, QFrame, QScrollArea,
    QGridLayout
)
import pyqtgraph as pg
import pyqtgraph.opengl as gl

import config
from tm16_decoder import TM16Decoder, TM16Frame
from short_edge_detector import ShortEdgeDetector, CoarseClusterResult
from cabin_detector import CabinDetector, CabinPositionResult
from cabin_measurement import measure_continuous_board_height, board_heights
from cabin_timeseries import CabinTimeSeriesWindow
from plc_adapter import compute_plc_measurements, PlcMeasurements

# Precompute colormap LUTs (256, 4)
TURBO_LUT = cm.turbo(np.linspace(0, 1, 256)).astype(np.float32)
VIRIDIS_LUT = cm.viridis(np.linspace(0, 1, 256)).astype(np.float32)
JET_LUT = cm.jet(np.linspace(0, 1, 256)).astype(np.float32)

# 16 distinct colors for laser rings 0..15
RING_COLORS = np.array([
    [0.92, 0.20, 0.20, 1.0],  # 0: Red
    [0.20, 0.85, 0.20, 1.0],  # 1: Green
    [0.25, 0.50, 0.95, 1.0],  # 2: Blue
    [0.98, 0.55, 0.10, 1.0],  # 3: Orange
    [0.75, 0.20, 0.85, 1.0],  # 4: Purple
    [0.10, 0.90, 0.90, 1.0],  # 5: Cyan
    [0.95, 0.30, 0.60, 1.0],  # 6: Pink
    [0.55, 0.80, 0.15, 1.0],  # 7: Lime
    [0.95, 0.85, 0.10, 1.0],  # 8: Yellow
    [0.20, 0.70, 0.50, 1.0],  # 9: Teal
    [0.65, 0.40, 0.25, 1.0],  # 10: Brown
    [0.45, 0.25, 0.75, 1.0],  # 11: Indigo
    [0.85, 0.50, 0.50, 1.0],  # 12: Coral
    [0.50, 0.85, 0.50, 1.0],  # 13: Mint
    [0.50, 0.70, 0.95, 1.0],  # 14: Sky Blue
    [1.00, 0.70, 0.85, 1.0],  # 15: Rose
], dtype=np.float32)


class PcapLoadWorker(QThread):
    """Background worker to decode PCAP frames without freezing GUI."""
    progress = Signal(float, str)
    finished = Signal(list, str)
    failed = Signal(str)

    def __init__(self, pcap_path: str, cut_deg: float = config.DEFAULT_CUT_ANGLE_DEG):
        super().__init__()
        self.pcap_path = pcap_path
        self.cut_deg = cut_deg
        self._is_cancelled = False
        self.decode_status = ''

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            started = time.perf_counter()
            decoder = TM16Decoder(cut_deg=self.cut_deg)

            def check_cancel(frac: float, msg: str):
                if self._is_cancelled:
                    raise InterruptedError("Loading cancelled")
                self.progress.emit(frac, msg)

            frames = decoder.decode_pcap(
                self.pcap_path,
                progress_callback=check_cancel
            )
            self.decode_status = f'{decoder.backend_status} | 解析耗时 {time.perf_counter() - started:.2f} 秒'
            if not self._is_cancelled:
                self.finished.emit(frames, self.pcap_path)
        except InterruptedError:
            pass
        except Exception as e:
            if not self._is_cancelled:
                self.failed.emit(str(e))


class TM16Visualizer(QMainWindow):
    def __init__(self, default_pcap: str = "", auto_load: bool = True):
        super().__init__()
        self.setWindowTitle("TM16 LiDAR 点云解析与算法验证平台 (临港项目)")
        self.resize(1580, 960)

        self.frames: list[TM16Frame] = []
        self.current_idx = 0
        self.is_playing = False
        self.use_site_coords = True

        # Short Edge detector instance with default config
        self.detector = ShortEdgeDetector()

        # Cabin Detector instance with ROI direct mapping
        c_cfg = getattr(config, 'CABIN_DETECTOR_CONFIG', {})
        self.cabin_detector = CabinDetector(
            roi=self.detector.roi,
            bin_size=c_cfg.get('bin_size', 0.17),
            min_peak_pts=c_cfg.get('min_peak_pts', 25),
            min_cabin_len=c_cfg.get('min_cabin_len', 6.00),
            max_cabin_len=c_cfg.get('max_cabin_len', 24.00),
            enable_z_uniformity=c_cfg.get('enable_z_uniformity', True),
            min_board_height=c_cfg.get('min_board_height', 1.80),
            max_empty_gap=c_cfg.get('max_empty_gap', 2.50),
            max_base_jump=c_cfg.get('max_base_jump', 1.00)
        )

        self.worker: PcapLoadWorker = None
        self.axis_items = []

        self.timeseries_win = CabinTimeSeriesWindow(None)
        self.timeseries_win.frame_selected.connect(self._on_timeseries_frame_selected)
        self.timeseries_win.btn_recompute.clicked.connect(self._trigger_timeseries_recompute)
        self.timeseries_win.results_ready.connect(self._update_current_frame_display)

        self._init_ui()
        self._init_timer()
        self._apply_dark_theme()


        self._create_axis_indicators()
        self._update_roi_box_geometry()

        if auto_load:
            if default_pcap and os.path.isfile(default_pcap):
                self.load_pcap(default_pcap)
            elif os.path.isfile(config.DEFAULT_TRUCK_PCAP):
                self.load_pcap(config.DEFAULT_TRUCK_PCAP)
            elif os.path.isfile(config.DEFAULT_EMPTY_GROUND_PCAP):
                self.load_pcap(config.DEFAULT_EMPTY_GROUND_PCAP)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        if hasattr(self, 'timeseries_win') and self.timeseries_win:
            self.timeseries_win.close()
        super().closeEvent(event)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 1. Top File Toolbar
        top_bar = QHBoxLayout()
        self.btn_open = QPushButton("📂 打开 PCAP 文件...")
        self.btn_open.clicked.connect(self._on_browse_file)
        top_bar.addWidget(self.btn_open)

        # Quick preset buttons
        self.btn_quick_truck = QPushButton("快速加载: 全流程空车.pcap")
        self.btn_quick_truck.clicked.connect(lambda: self.load_pcap(config.DEFAULT_TRUCK_PCAP))
        top_bar.addWidget(self.btn_quick_truck)

        self.btn_quick_ground = QPushButton("快速加载: 空地.pcap")
        self.btn_quick_ground.clicked.connect(lambda: self.load_pcap(config.DEFAULT_EMPTY_GROUND_PCAP))
        top_bar.addWidget(self.btn_quick_ground)

        self.lbl_pcap_info = QLabel("尚未加载 PCAP 文件")
        self.lbl_pcap_info.setStyleSheet("color: #aaa; font-weight: bold;")
        top_bar.addWidget(self.lbl_pcap_info, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(200)
        top_bar.addWidget(self.progress_bar)

        self.btn_timeseries = QPushButton("📈 车帮时序图（高度四图 / 位置车长三图）")
        self.btn_timeseries.setStyleSheet("background-color: #1e3a5f; color: #66b2ff; font-weight: bold; padding: 5px 12px;")
        self.btn_timeseries.setToolTip("打开高度四图及前车帮、后车帮、车长三图；共用时间轴与选帧，区分实测与预测")
        self.btn_timeseries.clicked.connect(self._open_timeseries_window)
        top_bar.addWidget(self.btn_timeseries)

        main_layout.addLayout(top_bar)

        # 2. Main Content Splitter (Left: 3D View, Right: Control & Stats)
        splitter = QSplitter(Qt.Horizontal)

        # 3D Viewport
        self.view3d = gl.GLViewWidget()
        self.view3d.setCameraPosition(distance=25, elevation=35, azimuth=45)
        self.view3d.setBackgroundColor('#1e1e24')

        # Grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(40, 40)
        self.grid.setSpacing(1, 1)
        self.view3d.addItem(self.grid)

        # Point cloud scatter item
        self.scatter = gl.GLScatterPlotItem()
        self.scatter.setGLOptions('opaque')
        self.view3d.addItem(self.scatter)

        # 1. 🟨 黄框：ROI 空间参数框 (Yellow)
        self.roi_box_item = gl.GLLinePlotItem()
        self.view3d.addItem(self.roi_box_item)

        # 2. 🟪 紫框：粗聚类车厢立体框 (Purple)
        self.cabin_box_item = gl.GLLinePlotItem()
        self.view3d.addItem(self.cabin_box_item)

        # 3. 🟥 红框：取车帮高的采样框 (Red - 前后车帮各一个)
        self.board_height_box_item = gl.GLLinePlotItem()
        self.view3d.addItem(self.board_height_box_item)

        # 车帮高连续立面散点 (高亮展示，无文字)
        self.board_height_pts_item = gl.GLScatterPlotItem()
        self.board_height_pts_item.setGLOptions('additive')
        self.view3d.addItem(self.board_height_pts_item)

        # 兼容旧引用：无文字
        self.coarse_box_item = self.cabin_box_item
        self.cabin_boards_item = gl.GLLinePlotItem()
        self.board_height_beam_item = gl.GLLinePlotItem()
        self.lbl_coarse_3d = gl.GLTextItem(pos=[0, 0, 0], text="")
        self.lbl_cabin_3d = gl.GLTextItem(pos=[0, 0, 0], text="")
        self.lbl_front_h_3d = gl.GLTextItem(pos=[0, 0, 0], text="")
        self.lbl_rear_h_3d = gl.GLTextItem(pos=[0, 0, 0], text="")

        splitter.addWidget(self.view3d)

        # Right Scroll Area for Parameter Tuning & Diagnostics
        # Right Scroll Area for Parameter Tuning & Diagnostics
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(480)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 4, 10, 4)
        right_layout.setSpacing(10)

        # =====================================================================
        # Card 1: 📊 实时识别结果与 PLC 寄存器
        # =====================================================================
        grp_diag = QGroupBox("📊 实时识别结果与 PLC 寄存器")
        vbox_diag = QVBoxLayout(grp_diag)
        vbox_diag.setSpacing(6)

        # Row 1: Status & Mask
        row_stat = QHBoxLayout()
        self.lbl_cabin_status = QLabel("检测状态: 尚未运行")
        self.lbl_cabin_status.setStyleSheet("color: #00e676; font-weight: bold; font-size: 12px;")
        self.lbl_plc_mask = QLabel("有效掩码: 0x00")
        self.lbl_plc_mask.setStyleSheet("color: #81d4fa; font-weight: bold; font-size: 12px;")
        row_stat.addWidget(self.lbl_cabin_status, 1)
        row_stat.addWidget(self.lbl_plc_mask)
        vbox_diag.addLayout(row_stat)

        # Row 2: 🟪 车厢定位 (紫框)
        self.lbl_coarse_x = QLabel("🟪 车厢范围 (紫框): - m | 净长: - m")
        self.lbl_coarse_x.setStyleSheet("color: #d187ff; font-weight: bold; font-family: Consolas, monospace; font-size: 12px;")
        vbox_diag.addWidget(self.lbl_coarse_x)

        # Row 3 & 4: 前车帮 (大X端 / 车头)
        self.lbl_front_pos = QLabel("前车帮 (大X): X = - m  | 车头 (Reg 43): - mm")
        self.lbl_front_pos.setStyleSheet("color: #ffffff; font-family: Consolas, monospace; font-size: 11px;")
        vbox_diag.addWidget(self.lbl_front_pos)

        self.lbl_front_h = QLabel("前车帮顶高:  Z = - m  | 前高 (Reg  1): - mm")
        self.lbl_front_h.setStyleSheet("color: #ffe082; font-family: Consolas, monospace; font-size: 11px;")
        vbox_diag.addWidget(self.lbl_front_h)

        # Row 5 & 6: 后车帮 (小X端 / 车尾)
        self.lbl_rear_pos = QLabel("后车帮 (小X): X = - m  | 车尾 (Reg 44): - mm")
        self.lbl_rear_pos.setStyleSheet("color: #ffffff; font-family: Consolas, monospace; font-size: 11px;")
        vbox_diag.addWidget(self.lbl_rear_pos)

        self.lbl_rear_h = QLabel("后车帮顶高:  Z = - m  | 后高 (Reg  7): - mm")
        self.lbl_rear_h.setStyleSheet("color: #ffe082; font-family: Consolas, monospace; font-size: 11px;")
        vbox_diag.addWidget(self.lbl_rear_h)

        self.lbl_front_info = self.lbl_front_pos
        self.lbl_rear_info = self.lbl_rear_pos

        # 兼容旧代码引用标签 (保持在后台更新)
        self.lbl_cabin_front = QLabel("")
        self.lbl_cabin_rear = QLabel("")
        self.lbl_cabin_len = QLabel("")
        self.lbl_cabin_size = QLabel("")
        self.lbl_cabin_zcheck = QLabel("")
        self.lbl_plc_head = QLabel("")
        self.lbl_plc_tail = QLabel("")
        self.lbl_plc_front_h = QLabel("")
        self.lbl_plc_rear_h = QLabel("")
        self.lbl_coarse_status = QLabel("")
        self.lbl_coarse_span = QLabel("")

        right_layout.addWidget(grp_diag)

        # =====================================================================
        # Card 2: 🟨 黄框：ROI 空间范围与地面过滤
        # =====================================================================
        grp_roi_tune = QGroupBox("🟨 黄框：ROI 空间范围 (工况过滤)")
        vbox_roi_tune = QVBoxLayout(grp_roi_tune)
        vbox_roi_tune.setSpacing(5)

        grid_roi = QGridLayout()
        grid_roi.setSpacing(5)

        grid_roi.addWidget(QLabel("X Min (前向):"), 0, 0)
        self.sp_roi_xmin = QDoubleSpinBox()
        self.sp_roi_xmin.setRange(-50.0, 50.0)
        self.sp_roi_xmin.setSingleStep(0.5)
        self.sp_roi_xmin.setValue(config.ROI_CONFIG['x_min'])
        self.sp_roi_xmin.setSuffix(" m")
        self.sp_roi_xmin.valueChanged.connect(self._on_roi_param_changed)
        grid_roi.addWidget(self.sp_roi_xmin, 0, 1)

        grid_roi.addWidget(QLabel("X Max (后向):"), 0, 2)
        self.sp_roi_xmax = QDoubleSpinBox()
        self.sp_roi_xmax.setRange(-50.0, 50.0)
        self.sp_roi_xmax.setSingleStep(0.5)
        self.sp_roi_xmax.setValue(config.ROI_CONFIG['x_max'])
        self.sp_roi_xmax.setSuffix(" m")
        self.sp_roi_xmax.valueChanged.connect(self._on_roi_param_changed)
        grid_roi.addWidget(self.sp_roi_xmax, 0, 3)

        grid_roi.addWidget(QLabel("Y Min (左侧):"), 1, 0)
        self.sp_roi_ymin = QDoubleSpinBox()
        self.sp_roi_ymin.setRange(-20.0, 20.0)
        self.sp_roi_ymin.setSingleStep(0.2)
        self.sp_roi_ymin.setValue(config.ROI_CONFIG['y_min'])
        self.sp_roi_ymin.setSuffix(" m")
        self.sp_roi_ymin.valueChanged.connect(self._on_roi_param_changed)
        grid_roi.addWidget(self.sp_roi_ymin, 1, 1)

        grid_roi.addWidget(QLabel("Y Max (右侧):"), 1, 2)
        self.sp_roi_ymax = QDoubleSpinBox()
        self.sp_roi_ymax.setRange(-20.0, 20.0)
        self.sp_roi_ymax.setSingleStep(0.2)
        self.sp_roi_ymax.setValue(config.ROI_CONFIG['y_max'])
        self.sp_roi_ymax.setSuffix(" m")
        self.sp_roi_ymax.valueChanged.connect(self._on_roi_param_changed)
        grid_roi.addWidget(self.sp_roi_ymax, 1, 3)

        grid_roi.addWidget(QLabel("Z Min (下限):"), 2, 0)
        self.sp_roi_zmin = QDoubleSpinBox()
        self.sp_roi_zmin.setRange(-5.0, 20.0)
        self.sp_roi_zmin.setSingleStep(0.1)
        self.sp_roi_zmin.setValue(config.ROI_CONFIG['z_min'])
        self.sp_roi_zmin.setSuffix(" m")
        self.sp_roi_zmin.valueChanged.connect(self._on_roi_param_changed)
        grid_roi.addWidget(self.sp_roi_zmin, 2, 1)

        grid_roi.addWidget(QLabel("Z Max (上限):"), 2, 2)
        self.sp_roi_zmax = QDoubleSpinBox()
        self.sp_roi_zmax.setRange(-5.0, 20.0)
        self.sp_roi_zmax.setSingleStep(0.2)
        self.sp_roi_zmax.setValue(config.ROI_CONFIG['z_max'])
        self.sp_roi_zmax.setSuffix(" m")
        self.sp_roi_zmax.valueChanged.connect(self._on_roi_param_changed)
        grid_roi.addWidget(self.sp_roi_zmax, 2, 3)

        grid_roi.addWidget(QLabel("地面滤除 Z:"), 3, 0)
        self.sp_ground_z = QDoubleSpinBox()
        self.sp_ground_z.setRange(-2.0, 5.0)
        self.sp_ground_z.setSingleStep(0.05)
        self.sp_ground_z.setValue(config.ROI_CONFIG.get('ground_z_threshold', 0.20))
        self.sp_ground_z.setSuffix(" m")
        self.sp_ground_z.valueChanged.connect(self._on_roi_param_changed)
        grid_roi.addWidget(self.sp_ground_z, 3, 1, 1, 3)

        vbox_roi_tune.addLayout(grid_roi)
        right_layout.addWidget(grp_roi_tune)

        # =====================================================================
        # Card 3: 🟪 紫框：粗聚类车厢参数 (圈车)
        # =====================================================================
        grp_cluster_tune = QGroupBox("🟪 紫框：粗聚类车厢参数 (圈车)")
        vbox_cluster_tune = QVBoxLayout(grp_cluster_tune)
        vbox_cluster_tune.setSpacing(5)

        grid_cluster = QGridLayout()
        grid_cluster.setSpacing(5)

        grid_cluster.addWidget(QLabel("X 轴划分网格数:"), 0, 0)
        self.sp_grid_num = QSpinBox()
        self.sp_grid_num.setRange(10, 1000)
        self.sp_grid_num.setSingleStep(10)
        self.sp_grid_num.setValue(200)
        self.sp_grid_num.setSuffix(" bin")
        self.sp_grid_num.valueChanged.connect(self._on_cluster_param_changed)
        grid_cluster.addWidget(self.sp_grid_num, 0, 1)

        self.lbl_grid_res = QLabel("分辨率: 0.268 m/格")
        self.lbl_grid_res.setStyleSheet("color: #88ccff; font-size: 11px;")
        grid_cluster.addWidget(self.lbl_grid_res, 0, 2, 1, 2)

        grid_cluster.addWidget(QLabel("单格有效点数:"), 1, 0)
        self.sp_min_pts = QSpinBox()
        self.sp_min_pts.setRange(1, 50)
        self.sp_min_pts.setSingleStep(1)
        self.sp_min_pts.setValue(3)
        self.sp_min_pts.setSuffix(" 点")
        self.sp_min_pts.valueChanged.connect(self._on_cluster_param_changed)
        grid_cluster.addWidget(self.sp_min_pts, 1, 1)

        grid_cluster.addWidget(QLabel("容忍最大空隙:"), 1, 2)
        self.sp_max_gap = QSpinBox()
        self.sp_max_gap.setRange(0, 20)
        self.sp_max_gap.setSingleStep(1)
        self.sp_max_gap.setValue(3)
        self.sp_max_gap.setSuffix(" 格")
        self.sp_max_gap.valueChanged.connect(self._on_cluster_param_changed)
        grid_cluster.addWidget(self.sp_max_gap, 1, 3)

        grid_cluster.addWidget(QLabel("最小整车跨度:"), 2, 0)
        self.sp_min_span = QSpinBox()
        self.sp_min_span.setRange(1, 50)
        self.sp_min_span.setSingleStep(1)
        self.sp_min_span.setValue(config.COARSE_CLUSTER_CONFIG.get('cluster_min_span', 6))
        self.sp_min_span.setSuffix(" 格")
        self.sp_min_span.valueChanged.connect(self._on_cluster_param_changed)
        grid_cluster.addWidget(self.sp_min_span, 2, 1)

        self.btn_reset_params = QPushButton("🔄 恢复默认配置")
        self.btn_reset_params.clicked.connect(self._reset_to_default_params)
        grid_cluster.addWidget(self.btn_reset_params, 2, 2, 1, 2)

        vbox_cluster_tune.addLayout(grid_cluster)
        right_layout.addWidget(grp_cluster_tune)

        # =====================================================================
        # Card 4: 🟥 红框：车帮高采样参数 (自底向上连续断档判定)
        # =====================================================================
        grp_height_tune = QGroupBox("🟥 红框：车帮高采样参数 (自底向上断档判定)")
        vbox_height_tune = QVBoxLayout(grp_height_tune)
        vbox_height_tune.setSpacing(5)

        grid_height = QGridLayout()
        grid_height.setSpacing(5)

        h_cfg = getattr(config, 'BOARD_HEIGHT_CONFIG', {})

        grid_height.addWidget(QLabel("采样半宽 (±X):"), 0, 0)
        self.sp_height_xmargin = QDoubleSpinBox()
        self.sp_height_xmargin.setRange(0.05, 1.50)
        self.sp_height_xmargin.setSingleStep(0.05)
        self.sp_height_xmargin.setValue(h_cfg.get('x_margin', 0.35))
        self.sp_height_xmargin.setSuffix(" m")
        self.sp_height_xmargin.valueChanged.connect(self._on_height_param_changed)
        grid_height.addWidget(self.sp_height_xmargin, 0, 1)

        grid_height.addWidget(QLabel("断档门限 (ΔZ):"), 0, 2)
        self.sp_height_maxgap = QDoubleSpinBox()
        self.sp_height_maxgap.setRange(0.05, 1.50)
        self.sp_height_maxgap.setSingleStep(0.05)
        self.sp_height_maxgap.setValue(h_cfg.get('max_z_gap', 0.35))
        self.sp_height_maxgap.setSuffix(" m")
        self.sp_height_maxgap.valueChanged.connect(self._on_height_param_changed)
        grid_height.addWidget(self.sp_height_maxgap, 0, 3)

        grid_height.addWidget(QLabel("最小连续点数:"), 1, 0)
        self.sp_height_minpts = QSpinBox()
        self.sp_height_minpts.setRange(1, 100)
        self.sp_height_minpts.setSingleStep(1)
        self.sp_height_minpts.setValue(h_cfg.get('min_continuous_pts', 5))
        self.sp_height_minpts.setSuffix(" 点")
        self.sp_height_minpts.valueChanged.connect(self._on_height_param_changed)
        grid_height.addWidget(self.sp_height_minpts, 1, 1)

        grid_height.addWidget(QLabel("顶部取值分位:"), 1, 2)
        self.sp_height_percentile = QDoubleSpinBox()
        self.sp_height_percentile.setRange(50.0, 100.0)
        self.sp_height_percentile.setSingleStep(0.5)
        self.sp_height_percentile.setValue(h_cfg.get('top_percentile', 98.0))
        self.sp_height_percentile.setSuffix(" %")
        self.sp_height_percentile.valueChanged.connect(self._on_height_param_changed)
        grid_height.addWidget(self.sp_height_percentile, 1, 3)

        self.btn_reset_height = QPushButton("🔄 恢复车帮高识别默认配置")
        self.btn_reset_height.clicked.connect(self._reset_height_params)
        grid_height.addWidget(self.btn_reset_height, 2, 0, 1, 4)

        vbox_height_tune.addLayout(grid_height)
        right_layout.addWidget(grp_height_tune)

        # =====================================================================
        # Card 5: 🎨 3D 视图与框体开关
        # =====================================================================
        grp_disp = QGroupBox("🎨 3D 视图与框体开关")
        vbox_disp = QVBoxLayout(grp_disp)
        vbox_disp.setSpacing(5)

        row_color = QHBoxLayout()
        row_color.addWidget(QLabel("着色模式:"))
        self.cb_color_mode = QComboBox()
        self.cb_color_mode.addItems([
            "按 Z 轴高度 (Height - Turbo)",
            "按 16 激光线束 (Laser Ring 0..15)",
            "按 反射强度 (Intensity)",
            "纯白单色 (Uniform)"
        ])
        self.cb_color_mode.currentIndexChanged.connect(self._update_current_frame_display)
        row_color.addWidget(self.cb_color_mode, 1)
        vbox_disp.addLayout(row_color)

        row_size = QHBoxLayout()
        row_size.addWidget(QLabel("点大小:"))
        self.slider_point_size = QSlider(Qt.Horizontal)
        self.slider_point_size.setRange(1, 10)
        self.slider_point_size.setValue(3)
        self.slider_point_size.valueChanged.connect(self._update_current_frame_display)
        row_size.addWidget(self.slider_point_size, 1)
        self.lbl_point_size = QLabel("3 px")
        row_size.addWidget(self.lbl_point_size)
        vbox_disp.addLayout(row_size)

        # 框体显隐开关 (直观对应颜色)
        self.chk_show_roi_box = QCheckBox("显示 🟨 ROI 参数框 (黄色)")
        self.chk_show_roi_box.setChecked(True)
        self.chk_show_roi_box.stateChanged.connect(self._on_roi_box_toggled)
        vbox_disp.addWidget(self.chk_show_roi_box)

        self.chk_show_cabin_box = QCheckBox("显示 🟪 粗聚类车厢框 (紫色)")
        self.chk_show_cabin_box.setChecked(True)
        self.chk_show_cabin_box.stateChanged.connect(self._update_current_frame_display)
        vbox_disp.addWidget(self.chk_show_cabin_box)

        self.chk_show_height_box = QCheckBox("显示 🟥 取车帮高采样框 (红色)")
        self.chk_show_height_box.setChecked(True)
        self.chk_show_height_box.stateChanged.connect(self._update_current_frame_display)
        vbox_disp.addWidget(self.chk_show_height_box)

        self.chk_show_grid = QCheckBox("显示 地面参考网格")
        self.chk_show_grid.setChecked(True)
        self.chk_show_grid.stateChanged.connect(self._on_grid_toggled)
        vbox_disp.addWidget(self.chk_show_grid)

        # 兼容旧开关属性
        self.chk_show_board_height = self.chk_show_height_box
        self.chk_show_coarse_box = self.chk_show_cabin_box
        self.cb_roi_mode = QComboBox()
        self.cb_roi_mode.addItems(["全部显示", "仅ROI", "全部"])
        self.chk_axis_dirs = QCheckBox("XYZ方向")
        self.lbl_stat_pts = QLabel("点数: 0")
        self.lbl_stat_time = QLabel("0.000s")
        self.lbl_stat_bounds = QLabel("")
        self.lbl_stat_calib = QLabel("")
        self.cb_coord = QComboBox()
        self.cb_coord.addItems(["现场外参坐标系", "雷达原生坐标系"])
        self.lbl_coord_formula = QLabel("")

        right_layout.addWidget(grp_disp)
        right_layout.addStretch(1)

        scroll_area.setWidget(right_panel)
        splitter.addWidget(scroll_area)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([950, 520])

        main_layout.addWidget(splitter, 1)

        # 3. Bottom Playback Controls
        bottom_box = QFrame()
        bottom_box.setFrameShape(QFrame.StyledPanel)
        bottom_layout = QHBoxLayout(bottom_box)
        bottom_layout.setContentsMargins(6, 6, 6, 6)
        bottom_layout.setSpacing(8)

        self.btn_prev = QPushButton("⏮ 上一帧")
        self.btn_prev.clicked.connect(self.prev_frame)
        bottom_layout.addWidget(self.btn_prev)

        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setFixedWidth(80)
        self.btn_play.clicked.connect(self.toggle_play)
        bottom_layout.addWidget(self.btn_play)

        self.btn_next = QPushButton("下一帧 ⏭")
        self.btn_next.clicked.connect(self.next_frame)
        bottom_layout.addWidget(self.btn_next)

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.valueChanged.connect(self._on_slider_moved)
        bottom_layout.addWidget(self.slider_frame, 1)

        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.valueChanged.connect(self._on_spin_changed)
        bottom_layout.addWidget(self.spin_frame)

        self.lbl_frame_counter = QLabel("0 / 0")
        self.lbl_frame_counter.setFixedWidth(75)
        self.lbl_frame_counter.setStyleSheet("font-weight: bold;")
        bottom_layout.addWidget(self.lbl_frame_counter)

        bottom_layout.addWidget(QLabel("速率:"))
        self.cb_fps = QComboBox()
        self.cb_fps.addItems(["1 FPS", "5 FPS", "10 FPS (标准)", "20 FPS", "30 FPS"])
        self.cb_fps.setCurrentIndex(2)
        self.cb_fps.currentIndexChanged.connect(self._on_fps_changed)
        bottom_layout.addWidget(self.cb_fps)

        self.chk_loop = QCheckBox("循环播放")
        self.chk_loop.setChecked(True)
        bottom_layout.addWidget(self.chk_loop)

        main_layout.addWidget(bottom_box)

    def _on_roi_param_changed(self):
        """Called whenever any ROI or Ground Z spinbox changes."""
        self.detector.roi['x_min'] = self.sp_roi_xmin.value()
        self.detector.roi['x_max'] = self.sp_roi_xmax.value()
        self.detector.roi['y_min'] = self.sp_roi_ymin.value()
        self.detector.roi['y_max'] = self.sp_roi_ymax.value()
        self.detector.roi['z_min'] = self.sp_roi_zmin.value()
        self.detector.roi['z_max'] = self.sp_roi_zmax.value()
        self.detector.ground_z_threshold = self.sp_ground_z.value()
        self.cabin_detector.roi = dict(self.detector.roi)
        self.timeseries_win.invalidate_results()
        if hasattr(self, 'timeseries_win') and self.timeseries_win.isVisible():
            self._trigger_timeseries_recompute()

        # Update resolution text
        x_span = max(0.01, self.detector.roi['x_max'] - self.detector.roi['x_min'])
        res = x_span / max(1, self.detector.y_grid_num)
        self.lbl_grid_res.setText(f"分辨率: {res:.3f} m/格")

        self._update_roi_box_geometry()
        self._update_current_frame_display()

    def _on_height_param_changed(self):
        """Called whenever board height continuity parameters change."""
        self.timeseries_win.invalidate_results()
        self._update_current_frame_display()

    def _reset_height_params(self):
        """Resets board height parameters to saved defaults from config.py."""
        h_cfg = getattr(config, 'BOARD_HEIGHT_CONFIG', {})
        self.sp_height_xmargin.blockSignals(True)
        self.sp_height_maxgap.blockSignals(True)
        self.sp_height_minpts.blockSignals(True)
        self.sp_height_percentile.blockSignals(True)

        self.sp_height_xmargin.setValue(h_cfg.get('x_margin', 0.30))
        self.sp_height_maxgap.setValue(h_cfg.get('max_z_gap', 0.35))
        self.sp_height_minpts.setValue(h_cfg.get('min_continuous_pts', 5))
        self.sp_height_percentile.setValue(h_cfg.get('top_percentile', 98.0))

        self.sp_height_xmargin.blockSignals(False)
        self.sp_height_maxgap.blockSignals(False)
        self.sp_height_minpts.blockSignals(False)
        self.sp_height_percentile.blockSignals(False)

        self._on_height_param_changed()

    def _on_cabin_param_changed(self):
        self._on_height_param_changed()

    def _reset_cabin_params(self):
        self._reset_height_params()

    def _on_cluster_param_changed(self):
        """Called whenever any clustering spinbox changes."""
        self.detector.y_grid_num = self.sp_grid_num.value()
        self.detector.cluster_min_points = self.sp_min_pts.value()
        self.detector.cluster_max_gap = self.sp_max_gap.value()
        self.detector.cluster_min_span = self.sp_min_span.value()

        # Update resolution text
        x_span = max(0.01, self.detector.roi['x_max'] - self.detector.roi['x_min'])
        res = x_span / max(1, self.detector.y_grid_num)
        self.lbl_grid_res.setText(f"分辨率: {res:.3f} m/格")

        self._update_current_frame_display()

    def _reset_to_default_params(self):
        """Restores all parameters to factory defaults from config.py."""
        self.sp_roi_xmin.blockSignals(True)
        self.sp_roi_xmax.blockSignals(True)
        self.sp_roi_ymin.blockSignals(True)
        self.sp_roi_ymax.blockSignals(True)
        self.sp_roi_zmin.blockSignals(True)
        self.sp_roi_zmax.blockSignals(True)
        self.sp_ground_z.blockSignals(True)
        self.sp_grid_num.blockSignals(True)
        self.sp_min_pts.blockSignals(True)
        self.sp_max_gap.blockSignals(True)
        self.sp_min_span.blockSignals(True)

        self.sp_roi_xmin.setValue(config.ROI_CONFIG['x_min'])
        self.sp_roi_xmax.setValue(config.ROI_CONFIG['x_max'])
        self.sp_roi_ymin.setValue(config.ROI_CONFIG['y_min'])
        self.sp_roi_ymax.setValue(config.ROI_CONFIG['y_max'])
        self.sp_roi_zmin.setValue(config.ROI_CONFIG['z_min'])
        self.sp_roi_zmax.setValue(config.ROI_CONFIG['z_max'])
        self.sp_ground_z.setValue(config.ROI_CONFIG.get('ground_z_threshold', 0.20))
        self.sp_grid_num.setValue(200)
        self.sp_min_pts.setValue(3)
        self.sp_max_gap.setValue(3)
        self.sp_min_span.setValue(config.COARSE_CLUSTER_CONFIG.get('cluster_min_span', 6))

        self.sp_roi_xmin.blockSignals(False)
        self.sp_roi_xmax.blockSignals(False)
        self.sp_roi_ymin.blockSignals(False)
        self.sp_roi_ymax.blockSignals(False)
        self.sp_roi_zmin.blockSignals(False)
        self.sp_roi_zmax.blockSignals(False)
        self.sp_ground_z.blockSignals(False)
        self.sp_grid_num.blockSignals(False)
        self.sp_min_pts.blockSignals(False)
        self.sp_max_gap.blockSignals(False)
        self.sp_min_span.blockSignals(False)

        self.cabin_detector.roi = dict(config.ROI_CONFIG)
        self._on_roi_param_changed()
        self._on_cluster_param_changed()

    def _create_axis_indicators(self):
        """Creates visible colored lines and labels for +X, -X, +Y, -Y, +Z, -Z."""
        L_pos = 6.0
        L_neg = 4.0

        red_pos = (1.0, 0.2, 0.2, 1.0)
        red_neg = (0.5, 0.1, 0.1, 0.7)
        green_pos = (0.2, 1.0, 0.2, 1.0)
        green_neg = (0.1, 0.5, 0.1, 0.7)
        blue_pos = (0.2, 0.5, 1.0, 1.0)
        blue_neg = (0.1, 0.2, 0.5, 0.7)

        axis_defs = [
            ('+X', [0, 0, 0], [L_pos, 0, 0], red_pos, [L_pos + 0.4, 0, 0], 3.0),
            ('-X', [0, 0, 0], [-L_neg, 0, 0], red_neg, [-L_neg - 0.7, 0, 0], 1.5),
            ('+Y', [0, 0, 0], [0, L_pos, 0], green_pos, [0, L_pos + 0.4, 0], 3.0),
            ('-Y', [0, 0, 0], [0, -L_neg, 0], green_neg, [0, -L_neg - 0.7, 0], 1.5),
            ('+Z', [0, 0, 0], [0, 0, L_pos], blue_pos, [0, 0, L_pos + 0.4], 3.0),
            ('-Z', [0, 0, 0], [0, 0, -L_neg], blue_neg, [0, 0, -L_neg - 0.7], 1.5),
        ]

        show_axis = self.chk_axis_dirs.isChecked()
        for name, p_start, p_end, col, text_pos, w in axis_defs:
            line_pts = np.array([p_start, p_end], dtype=np.float32)
            line_item = gl.GLLinePlotItem(pos=line_pts, mode='lines', color=col, width=w)
            line_item.setVisible(show_axis)
            self.view3d.addItem(line_item)
            text_item = gl.GLTextItem(pos=text_pos, text=name)
            text_item.setVisible(show_axis)
            self.view3d.addItem(text_item)
            self.axis_items.extend([line_item, text_item])

    def _on_axis_toggled(self, state: int):
        show = self.chk_axis_dirs.isChecked()
        for item in self.axis_items:
            item.setVisible(show)

    def _init_timer(self):
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_step)
        self.play_fps = 10
        self.play_timer.setInterval(100)

    def _apply_dark_theme(self):
        style = """
        QMainWindow { background-color: #1a1a1f; color: #e0e0e0; }
        QWidget { background-color: #1a1a1f; color: #e0e0e0; font-family: 'Segoe UI', Arial; }
        QGroupBox {
            border: 1px solid #33333d; border-radius: 6px;
            margin-top: 10px; font-weight: bold; padding-top: 10px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #66b2ff; }
        QPushButton {
            background-color: #2b2b36; border: 1px solid #444455;
            border-radius: 4px; padding: 5px 12px; font-weight: 500;
        }
        QPushButton:hover { background-color: #3b3b4d; border-color: #66b2ff; }
        QPushButton:pressed { background-color: #1a1a24; }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #2b2b36; border: 1px solid #444455;
            border-radius: 4px; padding: 3px 6px; color: #ffffff;
        }
        QSlider::groove:horizontal {
            border: 1px solid #333340; height: 6px; background: #262633; border-radius: 3px;
        }
        QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 3px; }
        QSlider::handle:horizontal {
            background: #ffffff; border: 1px solid #3b82f6; width: 14px;
            margin-top: -4px; margin-bottom: -4px; border-radius: 7px;
        }
        QTableWidget {
            background-color: #22222b; border: 1px solid #333340;
            gridline-color: #2d2d3a; selection-background-color: #3b82f6;
        }
        QHeaderView::section {
            background-color: #2b2b36; border: 1px solid #333340;
            padding: 4px; font-weight: bold; color: #cccccc;
        }
        QProgressBar {
            border: 1px solid #333340; border-radius: 4px; text-align: center; background: #22222b;
        }
        QProgressBar::chunk { background-color: #3b82f6; border-radius: 3px; }
        QScrollBar:vertical {
            background: #1a1a1f; width: 8px; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #444455; min-height: 20px; border-radius: 4px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """
        self.setStyleSheet(style)

    def _on_browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 TM16 PCAP 抓包文件",
            config.DEFAULT_PCAP_DIR,
            "PCAP 抓包 (*.pcap *.pcapng);;所有文件 (*.*)"
        )
        if path:
            self.load_pcap(path)

    def load_pcap(self, pcap_path: str):
        if not os.path.isfile(pcap_path):
            QMessageBox.warning(self, "错误", f"找不到文件:\n{pcap_path}")
            return

        if self.is_playing:
            self.toggle_play()

        self.btn_open.setEnabled(False)
        self.btn_quick_ground.setEnabled(False)
        self.btn_quick_truck.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_pcap_info.setText(f"正在加载: {os.path.basename(pcap_path)}...")

        self.worker = PcapLoadWorker(pcap_path)
        self.worker.progress.connect(self._on_load_progress)
        self.worker.finished.connect(self._on_load_finished)
        self.worker.failed.connect(self._on_load_failed)
        self.worker.start()

    @Slot(float, str)
    def _on_load_progress(self, frac: float, msg: str):
        self.progress_bar.setValue(int(frac * 100))
        self.lbl_pcap_info.setText(msg)

    @Slot(list, str)
    def _on_load_finished(self, frames: list, pcap_path: str):
        self.timeseries_win.invalidate_results()
        self.frames = frames
        self.btn_open.setEnabled(True)
        self.btn_quick_ground.setEnabled(True)
        self.btn_quick_truck.setEnabled(True)
        self.progress_bar.setVisible(False)

        total = len(self.frames)
        file_name = os.path.basename(pcap_path)
        decode_status = self.worker.decode_status if self.worker else ''
        self.lbl_pcap_info.setText(f"已加载: {file_name} | 共计 {total} 帧数据 | {decode_status}")

        if total > 0:
            self.slider_frame.setRange(0, total - 1)
            self.spin_frame.setRange(0, total - 1)
            self.current_idx = 0
            self.slider_frame.setValue(0)
            self.spin_frame.setValue(0)
            self._update_current_frame_display()
            if hasattr(self, 'timeseries_win') and self.timeseries_win.isVisible():
                self._trigger_timeseries_recompute()
        else:
            QMessageBox.information(self, "提示", "PCAP 文件中未解析到完整的激光雷达帧。")

    @Slot(str)
    def _on_load_failed(self, err_msg: str):
        self.btn_open.setEnabled(True)
        self.btn_quick_ground.setEnabled(True)
        self.btn_quick_truck.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_pcap_info.setText("解析失败")
        QMessageBox.critical(self, "解析出错", f"解析 PCAP 异常:\n{err_msg}")

    def _on_coord_changed(self, idx: int):
        self.use_site_coords = (idx == 0)
        if self.use_site_coords:
            self.lbl_coord_formula.setText("现场坐标: X=系统Y, Y=系统Z, Z=9.4-系统X")
        else:
            self.lbl_coord_formula.setText("雷达原生: X=正前, Y=负左, Z=垂直上")
        self._update_roi_box_geometry()
        self._update_current_frame_display()

    def _on_roi_box_toggled(self, state: int):
        if hasattr(self, 'chk_show_roi_box'):
            self.roi_box_item.setVisible(self.chk_show_roi_box.isChecked())

    def _on_grid_toggled(self, state: int):
        show = self.chk_show_grid.isChecked()
        self.grid.setVisible(show)

    def _update_roi_box_geometry(self):
        """🟨 黄框：ROI 空间参数框 (无文字)"""
        if not hasattr(self, 'chk_show_roi_box'):
            return

        if not self.use_site_coords or not self.chk_show_roi_box.isChecked():
            self.roi_box_item.setVisible(False)
            return

        roi = self.detector.roi
        x0, x1 = float(roi['x_min']), float(roi['x_max'])
        y0, y1 = float(roi['y_min']), float(roi['y_max'])
        z0, z1 = float(roi['z_min']), float(roi['z_max'])

        edges = [
            # Bottom face
            [x0, y0, z0], [x1, y0, z0],
            [x1, y0, z0], [x1, y1, z0],
            [x1, y1, z0], [x0, y1, z0],
            [x0, y1, z0], [x0, y0, z0],
            # Top face
            [x0, y0, z1], [x1, y0, z1],
            [x1, y0, z1], [x1, y1, z1],
            [x1, y1, z1], [x0, y1, z1],
            [x0, y1, z1], [x0, y0, z1],
            # 4 vertical pillars
            [x0, y0, z0], [x0, y0, z1],
            [x1, y0, z0], [x1, y0, z1],
            [x1, y1, z0], [x1, y1, z1],
            [x0, y1, z0], [x0, y1, z1],
        ]
        pts = np.array(edges, dtype=np.float32)
        self.roi_box_item.setData(pos=pts, mode='lines', color=(1.0, 0.85, 0.0, 0.95), width=2.0)
        self.roi_box_item.setVisible(True)

    def _draw_cabin_box(self, coarse_x: Optional[Tuple[float, float]], zf_raw: Optional[float] = None, zr_raw: Optional[float] = None):
        """🟪 紫色框：粗聚类的框，相当于车厢框 (无文字)"""
        if (not self.use_site_coords) or (not hasattr(self, 'chk_show_cabin_box')) or (not self.chk_show_cabin_box.isChecked()) or (coarse_x is None):
            self.cabin_box_item.setVisible(False)
            return

        x0 = float(min(coarse_x[0], coarse_x[1]))
        x1 = float(max(coarse_x[0], coarse_x[1]))
        roi = self.detector.roi
        y0, y1 = float(roi['y_min']), float(roi['y_max'])
        z0 = float(roi['z_min'])
        z1 = float(roi['z_max'])

        zf = zf_raw if (zf_raw is not None and math.isfinite(zf_raw)) else z1
        zr = zr_raw if (zr_raw is not None and math.isfinite(zr_raw)) else z1

        # 12 edges for the full 3D cabin box (Purple)
        edges = [
            # Bottom face
            [x0, y0, z0], [x1, y0, z0],
            [x1, y0, z0], [x1, y1, z0],
            [x1, y1, z0], [x0, y1, z0],
            [x0, y1, z0], [x0, y0, z0],
            # Top face (connecting measured heights)
            [x0, y0, zr], [x1, y0, zf],
            [x1, y0, zf], [x1, y1, zf],
            [x1, y1, zf], [x0, y1, zr],
            [x0, y1, zr], [x0, y0, zr],
            # 4 vertical pillars
            [x0, y0, z0], [x0, y0, zr],
            [x1, y0, z0], [x1, y0, zf],
            [x1, y1, z0], [x1, y1, zf],
            [x0, y1, z0], [x0, y1, zr],
        ]
        pts_box = np.array(edges, dtype=np.float32)
        self.cabin_box_item.setData(pos=pts_box, mode='lines', color=(0.78, 0.20, 1.0, 1.0), width=3.0)
        self.cabin_box_item.setVisible(True)

    def _draw_coarse_box(self, coarse_x: Optional[Tuple[float, float]]):
        """别名兼容：由 _draw_cabin_box 统一绘制紫色车厢框"""
        pass

    def _draw_board_height_boxes(self, cabin_res: CabinPositionResult,
                                 zf_raw: Optional[float] = None, zr_raw: Optional[float] = None,
                                 zf_pts_arr: Optional[np.ndarray] = None, zr_pts_arr: Optional[np.ndarray] = None):
        """🟥 红色框：取车帮高的框 (前后车帮采样范围各画一个红色立体框，无文字)"""
        if (not self.use_site_coords) or (not hasattr(self, 'chk_show_height_box')) or (not self.chk_show_height_box.isChecked()) or (cabin_res.status != "ok") or (cabin_res.x_front is None) or (cabin_res.x_rear is None):
            self.board_height_box_item.setVisible(False)
            self.board_height_pts_item.setVisible(False)
            return

        roi = self.detector.roi
        y0, y1 = float(roi['y_min']), float(roi['y_max'])
        z0 = float(roi['z_min'])
        z_default = float(roi['z_max'])

        h_cfg = getattr(config, 'BOARD_HEIGHT_CONFIG', {})
        x_margin = self.sp_height_xmargin.value() if hasattr(self, 'sp_height_xmargin') else h_cfg.get('x_margin', 0.35)

        zf = zf_raw if (zf_raw is not None and math.isfinite(zf_raw)) else z_default
        zr = zr_raw if (zr_raw is not None and math.isfinite(zr_raw)) else z_default

        edges = []

        # 1. 前车帮采样红框 (大X端)
        xf = cabin_res.x_front
        xf0, xf1 = xf - x_margin, xf + x_margin
        edges.extend([
            # Bottom
            [xf0, y0, z0], [xf1, y0, z0],
            [xf1, y0, z0], [xf1, y1, z0],
            [xf1, y1, z0], [xf0, y1, z0],
            [xf0, y1, z0], [xf0, y0, z0],
            # Top
            [xf0, y0, zf], [xf1, y0, zf],
            [xf1, y0, zf], [xf1, y1, zf],
            [xf1, y1, zf], [xf0, y1, zf],
            [xf0, y1, zf], [xf0, y0, zf],
            # Pillars
            [xf0, y0, z0], [xf0, y0, zf],
            [xf1, y0, z0], [xf1, y0, zf],
            [xf1, y1, z0], [xf1, y1, zf],
            [xf0, y1, z0], [xf0, y1, zf],
        ])

        # 2. 后车帮采样红框 (小X端)
        xr = cabin_res.x_rear
        xr0, xr1 = xr - x_margin, xr + x_margin
        edges.extend([
            # Bottom
            [xr0, y0, z0], [xr1, y0, z0],
            [xr1, y0, z0], [xr1, y1, z0],
            [xr1, y1, z0], [xr0, y1, z0],
            [xr0, y1, z0], [xr0, y0, z0],
            # Top
            [xr0, y0, zr], [xr1, y0, zr],
            [xr1, y0, zr], [xr1, y1, zr],
            [xr1, y1, zr], [xr0, y1, zr],
            [xr0, y1, zr], [xr0, y0, zr],
            # Pillars
            [xr0, y0, z0], [xr0, y0, zr],
            [xr1, y0, z0], [xr1, y0, zr],
            [xr1, y1, z0], [xr1, y1, zr],
            [xr0, y1, z0], [xr0, y1, zr],
        ])

        pts_boxes = np.array(edges, dtype=np.float32)
        self.board_height_box_item.setData(pos=pts_boxes, mode='lines', color=(1.0, 0.15, 0.15, 1.0), width=2.5)
        self.board_height_box_item.setVisible(True)

        # 3. 显示连续立面提取点云 (黄色散点，无字)
        pts_list = []
        if zf_pts_arr is not None and len(zf_pts_arr) > 0:
            pts_list.append(zf_pts_arr[:, :3])
        if zr_pts_arr is not None and len(zr_pts_arr) > 0:
            pts_list.append(zr_pts_arr[:, :3])

        if len(pts_list) > 0:
            all_wall_pts = np.vstack(pts_list).astype(np.float32)
            self.board_height_pts_item.setData(
                pos=all_wall_pts,
                color=(1.0, 0.90, 0.2, 0.95),
                size=float(self.slider_point_size.value() + 2),
                pxMode=True
            )
            self.board_height_pts_item.setVisible(True)
        else:
            self.board_height_pts_item.setVisible(False)

    def _draw_board_height_visuals(self, *args, **kwargs):
        """别名兼容"""
        self._draw_board_height_boxes(*args, **kwargs)

    def _update_current_frame_display(self):
        if not self.frames or self.current_idx >= len(self.frames):
            self.scatter.setData(pos=np.empty((0, 3)))
            self._draw_coarse_box(None)
            self._draw_cabin_box(CabinPositionResult(status="no_points"))
            return

        frame = self.frames[self.current_idx]
        pts_all = frame.get_points(self.use_site_coords)

        # Update frame counter label
        self.lbl_frame_counter.setText(f"{self.current_idx + 1} / {len(self.frames)}")

        # 1. Filter ROI points
        roi_pts = self.detector.filter_roi(frame.points_site)
        roi_count = len(roi_pts)

        # 2. Run Step 2: Coarse clustering to bound vehicle
        coarse_res: CoarseClusterResult = self.detector.detect_coarse(frame.points_site)
        if coarse_res.status == "ok" and coarse_res.coarse_x is not None:
            c_min, c_max = coarse_res.coarse_x
            self.lbl_coarse_status.setText("粗检状态: ✅ 检出粗范围")
            self.lbl_coarse_status.setStyleSheet("color: #00e676; font-weight: bold;")
            self.lbl_coarse_x.setText(f"整车 X 粗范围: [{c_min:6.2f}, {c_max:6.2f}] m")
            self.lbl_coarse_span.setText(f"整车跨度: {coarse_res.cluster_span:.2f} m")
            self._draw_coarse_box(coarse_res.coarse_x)
        else:
            self.lbl_coarse_status.setText(f"粗检状态: ⚪ 未检出 ({coarse_res.status})")
            self.lbl_coarse_status.setStyleSheet("color: #888; font-weight: bold;")
            self.lbl_coarse_x.setText("整车 X 粗范围: -")
            self.lbl_coarse_span.setText("整车跨度: 0.00 m")
            self._draw_coarse_box(None)

        # 3. Cabin Detection: 以粗聚类圈车结果作为前后车帮 (若时序未就绪则使用瞬时粗聚类)
        tracked = self.timeseries_win.position_results
        if len(tracked) == len(self.frames):
            cabin_res: CabinPositionResult = tracked[self.current_idx]
        else:
            cabin_res: CabinPositionResult = self.cabin_detector.detect(roi_pts, roi=self.detector.roi, coarse_res=coarse_res)

        # 2.5 Compute continuous board height
        zf_raw, zr_raw = None, None
        zf_pts, zr_pts = 0, 0
        zf_pts_arr, zr_pts_arr = np.empty((0, 3)), np.empty((0, 3))
        if cabin_res.status == "ok" and cabin_res.x_front is not None and cabin_res.x_rear is not None:
            h_cfg = getattr(config, 'BOARD_HEIGHT_CONFIG', {})
            x_margin = self.sp_height_xmargin.value() if hasattr(self, 'sp_height_xmargin') else h_cfg.get('x_margin', 0.35)
            max_gap = self.sp_height_maxgap.value() if hasattr(self, 'sp_height_maxgap') else h_cfg.get('max_z_gap', 0.35)
            min_pts = self.sp_height_minpts.value() if hasattr(self, 'sp_height_minpts') else h_cfg.get('min_continuous_pts', 5)
            pct = self.sp_height_percentile.value() if hasattr(self, 'sp_height_percentile') else h_cfg.get('top_percentile', 98.0)

            # 车帮测高使用 frame.points_site (完整点云，不受 ROI 的 z_max 裁剪)
            zf_raw, zf_pts, _, zf_pts_arr = measure_continuous_board_height(
                frame.points_site, cabin_res.x_front,
                x_margin=x_margin,
                z_min=h_cfg.get('z_min', 0.50),
                z_max=h_cfg.get('z_max', 20.00),
                y_min=self.detector.roi['y_min'],
                y_max=self.detector.roi['y_max'],
                max_z_gap=max_gap,
                min_continuous_pts=min_pts,
                top_percentile=pct
            )
            zr_raw, zr_pts, _, zr_pts_arr = measure_continuous_board_height(
                frame.points_site, cabin_res.x_rear,
                x_margin=x_margin,
                z_min=h_cfg.get('z_min', 0.50),
                z_max=h_cfg.get('z_max', 20.00),
                y_min=self.detector.roi['y_min'],
                y_max=self.detector.roi['y_max'],
                max_z_gap=max_gap,
                min_continuous_pts=min_pts,
                top_percentile=pct
            )

        # 2.6 Compute PLC 4-field measurements
        plc_meas = compute_plc_measurements(
            cabin_res.x_front, cabin_res.x_rear, zf_raw, zr_raw, cabin_res.status
        )

        if cabin_res.status == "ok" and cabin_res.x_front is not None and cabin_res.x_rear is not None:
            event_text = {'resumed': '同一目标恢复', 'reacquired': '目标重新捕获',
                          'tracked': '持续跟踪',
                          'local_reacquired': '缺点后恢复两端立面'}.get(cabin_res.track_event, 'X轴粗聚类圈车定位')
            self.lbl_cabin_status.setText(f"检测状态: ✅ {event_text}")
            self.lbl_cabin_status.setStyleSheet("color: #00e676; font-weight: bold; font-size: 12px;")
            self.lbl_coarse_x.setText(f"🟪 车厢范围 (紫框): [{cabin_res.x_rear:6.2f}, {cabin_res.x_front:6.2f}] m | 跨度: {cabin_res.cabin_length:.2f} m")

            fh_str = f"{plc_meas.front_height_mm:5d} mm" if (zf_raw is not None and plc_meas.is_front_h_valid) else "缺测"
            rh_str = f"{plc_meas.rear_height_mm:5d} mm" if (zr_raw is not None and plc_meas.is_rear_h_valid) else "缺测"
            zf_str = f"{zf_raw:.3f} m" if zf_raw is not None else "缺测"
            zr_str = f"{zr_raw:.3f} m" if zr_raw is not None else "缺测"

            self.lbl_front_pos.setText(f"前车帮 (大X): X = {cabin_res.x_front:6.2f} m  | 车头 (Reg 43): {plc_meas.head_position_mm:6d} mm")
            self.lbl_front_h.setText(f"前车帮顶高:  Z = {zf_str}  | 前高 (Reg  1): {fh_str}")

            self.lbl_rear_pos.setText(f"后车帮 (小X): X = {cabin_res.x_rear:6.2f} m  | 车尾 (Reg 44): {plc_meas.tail_position_mm:6d} mm")
            self.lbl_rear_h.setText(f"后车帮顶高:  Z = {zr_str}  | 后高 (Reg  7): {rh_str}")
            self.lbl_plc_mask.setText(f"有效掩码: 0x{plc_meas.valid_mask:02X}")

            # 绘制 🟪 紫色框 (粗聚类车厢框)
            self._draw_cabin_box((cabin_res.x_rear, cabin_res.x_front), zf_raw, zr_raw)
            # 绘制 🟥 红色框 (前后车帮采样框)
            self._draw_board_height_boxes(cabin_res, zf_raw, zr_raw, zf_pts_arr, zr_pts_arr)
        elif cabin_res.status == 'predicted':
            self.lbl_cabin_status.setText('检测状态: ⚠️ 缺点短时预测，当前位置无本帧完整实测')
            self.lbl_cabin_status.setStyleSheet('color: #fbbf24; font-weight: bold; font-size: 12px;')
            self.lbl_coarse_x.setText(f'预测范围: [{cabin_res.x_rear:.2f}, {cabin_res.x_front:.2f}] m | 预测跨度: {cabin_res.cabin_length:.2f} m')
            self.lbl_front_pos.setText(f'前车帮预测 X = {cabin_res.x_front:.2f} m | 车头 (Reg 43): 无效')
            self.lbl_rear_pos.setText(f'后车帮预测 X = {cabin_res.x_rear:.2f} m | 车尾 (Reg 44): 无效')
            self.lbl_front_h.setText('前车帮顶高: 本帧完整测量无效，请查看时序来源')
            self.lbl_rear_h.setText('后车帮顶高: 本帧完整测量无效，请查看时序来源')
            self.lbl_plc_mask.setText('有效掩码: 0x00（位置预测）')
            self._draw_cabin_box((cabin_res.x_rear, cabin_res.x_front), None, None)
            self._draw_board_height_boxes(cabin_res, None, None, np.empty((0, 3)), np.empty((0, 3)))
        else:
            self.lbl_cabin_status.setText(f"检测状态: ⚪ 未检出车厢 ({cabin_res.status})")
            self.lbl_cabin_status.setStyleSheet("color: #ff9800; font-weight: bold; font-size: 12px;")
            self.lbl_coarse_x.setText("🟪 车厢范围 (紫框): - m | 净长: - m")
            self.lbl_front_pos.setText("前车帮 (大X): X = - m  | 车头 (Reg 43): - mm")
            self.lbl_front_h.setText("前车帮顶高:  Z = - m  | 前高 (Reg  1): - mm")
            self.lbl_rear_pos.setText("后车帮 (小X): X = - m  | 车尾 (Reg 44): - mm")
            self.lbl_rear_h.setText("后车帮顶高:  Z = - m  | 后高 (Reg  7): - mm")
            self.lbl_plc_mask.setText("有效掩码: 0x00")
            self._draw_cabin_box(None, None, None)
            self._draw_board_height_boxes(cabin_res, None, None, np.empty((0, 3)), np.empty((0, 3)))

        # 绘制 🟨 黄色框 (ROI 空间框)
        self._update_roi_box_geometry()

        # 4. Handle ROI View Mode
        roi = self.detector.roi
        if self.use_site_coords:
            x_all, y_all, z_all = pts_all[:, 0], pts_all[:, 1], pts_all[:, 2]
            roi_mask = (
                np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(z_all) &\
                (x_all > roi['x_min']) & (x_all < roi['x_max']) &\
                (y_all > roi['y_min']) & (y_all < roi['y_max']) &\
                (z_all > roi['z_min']) & (z_all < roi['z_max']) &\
                (z_all > self.detector.ground_z_threshold)
            )
        else:
            roi_mask = np.ones(len(pts_all), dtype=bool)

        roi_mode = self.cb_roi_mode.currentIndex()

        if roi_mode == 1:
            display_pts = pts_all[roi_mask]
            is_roi_display = np.ones(len(display_pts), dtype=bool)
        else:
            display_pts = pts_all
            is_roi_display = roi_mask

        N = len(display_pts)
        self.lbl_stat_pts.setText(f"总点数: {len(pts_all):,} | ROI 点数: {roi_count:,}")
        self.lbl_stat_time.setText(f"时间戳: {frame.timestamp:.4f} s | 报文包数: {frame.packet_count}")
        self.lbl_stat_calib.setText(f"标定状态: {frame.calibration_state}")

        # Update angle table
        if hasattr(self, 'table_angles'):
            for r in range(16):
                item = self.table_angles.item(r, 1)
                if item:
                    item.setText(f"{frame.vertical_angles[r]:.3f}°")

        if N == 0:
            self.scatter.setData(pos=np.empty((0, 3)))
            self.lbl_stat_bounds.setText("点云范围 (XYZ): 无有效点")
            return

        xyz = display_pts[:, :3]
        intensities = display_pts[:, 3]
        rings = display_pts[:, 4].astype(int)

        # Update bounds text
        min_x, max_x = xyz[:, 0].min(), xyz[:, 0].max()
        min_y, max_y = xyz[:, 1].min(), xyz[:, 1].max()
        min_z, max_z = xyz[:, 2].min(), xyz[:, 2].max()
        self.lbl_stat_bounds.setText(
            f"点云范围 (XYZ):\n"
            f" X: [{min_x:6.2f}, {max_x:6.2f}] m\n"
            f" Y: [{min_y:6.2f}, {max_y:6.2f}] m\n"
            f" Z: [{min_z:6.2f}, {max_z:6.2f}] m"
        )

        # Colormap computation
        color_mode = self.cb_color_mode.currentIndex()
        if color_mode == 0: # Height Turbo
            z_vals = xyz[:, 2]
            z_norm = np.clip((z_vals - min_z) / (max_z - min_z + 1e-6), 0.0, 1.0)
            lut_indices = (z_norm * 255).astype(int)
            colors = TURBO_LUT[lut_indices].copy()
        elif color_mode == 1: # Laser Ring 16
            ring_clamped = np.clip(rings, 0, 15)
            colors = RING_COLORS[ring_clamped].copy()
        elif color_mode == 2: # Intensity
            norm_i = np.clip(intensities / 255.0, 0.0, 1.0)
            lut_indices = (norm_i * 255).astype(int)
            colors = JET_LUT[lut_indices].copy()
        else: # Uniform
            colors = np.ones((N, 4), dtype=np.float32)
            colors[:, :3] = 0.95

        # If Highlight Mode (0), set outside points to crisp pure white
        if roi_mode == 0 and self.use_site_coords:
            outside_mask = ~is_roi_display
            if np.any(outside_mask):
                colors[outside_mask] = np.array([1.0, 1.0, 1.0, 0.85], dtype=np.float32)

        pt_size = self.slider_point_size.value()
        self.lbl_point_size.setText(f"{pt_size} px")
        self.scatter.setData(pos=xyz, color=colors, size=pt_size)

        if hasattr(self, 'timeseries_win') and self.timeseries_win.isVisible():
            self.timeseries_win.set_current_frame(self.current_idx)

    def _on_slider_moved(self, val: int):
        if val != self.current_idx and val < len(self.frames):
            self.current_idx = val
            self.spin_frame.blockSignals(True)
            self.spin_frame.setValue(val)
            self.spin_frame.blockSignals(False)
            self._update_current_frame_display()

    def _on_spin_changed(self, val: int):
        if val != self.current_idx and val < len(self.frames):
            self.current_idx = val
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(val)
            self.slider_frame.blockSignals(False)
            self._update_current_frame_display()

    def prev_frame(self):
        if self.current_idx > 0:
            self.slider_frame.setValue(self.current_idx - 1)

    def next_frame(self):
        if self.current_idx < len(self.frames) - 1:
            self.slider_frame.setValue(self.current_idx + 1)
        elif self.chk_loop.isChecked() and len(self.frames) > 0:
            self.slider_frame.setValue(0)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("⏸ 暂停")
            self.play_timer.start()
        else:
            self.btn_play.setText("▶ 播放")
            self.play_timer.stop()

    def _on_play_step(self):
        if not self.frames:
            return
        if self.current_idx < len(self.frames) - 1:
            self.slider_frame.setValue(self.current_idx + 1)
        else:
            if self.chk_loop.isChecked():
                self.slider_frame.setValue(0)
            else:
                self.toggle_play()

    def _on_fps_changed(self, idx: int):
        fps_map = {0: 1, 1: 5, 2: 10, 3: 20, 4: 30}
        self.play_fps = fps_map.get(idx, 10)
        interval = int(1000.0 / self.play_fps)
        self.play_timer.setInterval(interval)


    def _open_timeseries_window(self):
        self.timeseries_win.show()
        self.timeseries_win.raise_()
        self.timeseries_win.activateWindow()
        # 打开时若当前没有计算结果或帧数不匹配，自动秒级开始异步计算呈现最新曲线！
        if (not self.timeseries_win.points_data or len(self.timeseries_win.position_results) != len(self.frames)) and self.frames:
            self._trigger_timeseries_recompute()
        else:
            self.timeseries_win.set_current_frame(self.current_idx)

    def _trigger_timeseries_recompute(self):
        if self.frames:
            height_config = dict(config.BOARD_HEIGHT_CONFIG,
                                 x_margin=self.sp_height_xmargin.value(),
                                 max_z_gap=self.sp_height_maxgap.value(),
                                 min_continuous_pts=self.sp_height_minpts.value(),
                                 top_percentile=self.sp_height_percentile.value())
            self.timeseries_win.start_computation(self.frames, self.detector, self.cabin_detector,
                                                 height_config=height_config)

    def _on_timeseries_frame_selected(self, frame_idx: int):
        if 0 <= frame_idx < len(self.frames):
            self.slider_frame.setValue(frame_idx)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    pcap_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    window = TM16Visualizer(pcap_arg, auto_load=True)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
