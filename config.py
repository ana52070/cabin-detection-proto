# -*- coding: utf-8 -*-
import numpy as np
import os

# Default Project PCAP paths
PROJECT_ROOT = r"D:\imformation\a企业实习\项目任务\26年10月\LD_MLScan_common(嘉兴_临港_营口)"
DEFAULT_PCAP_DIR = os.path.join(PROJECT_ROOT, "pcap")
DEFAULT_EMPTY_GROUND_PCAP = os.path.join(DEFAULT_PCAP_DIR, "空地.pcap")
DEFAULT_TRUCK_PCAP = os.path.join(DEFAULT_PCAP_DIR, "全流程空车.pcap")

# Network Ports
MSOP_PORT = 2368
DIFOP_PORT = 8603

# Protocol constants
MSOP_PACKET_SIZE = 1206
DIFOP_PACKET_SIZE = 1206
BLOCKS_PER_PACKET = 12
BLOCK_SIZE = 100
LASER_COUNT = 16
FIRINGS_PER_BLOCK = 2
DISTANCE_RES = 0.005 # 5mm per LSB

TM16_BLOCK_TDURATION_US = 98.304
TM16_DSR_TOFFSET_US = 3.072
TM16_FIRING_TOFFSET_US = 49.152

DEFAULT_VERTICAL_ANGLES = np.array([
    -15.0,   1.0,  -13.0,   3.0,
    -11.0,   5.0,   -9.0,   7.0,
     -7.0,   9.0,   -5.0,  11.0,
     -3.0,  13.0,   -1.0,  15.0
], dtype=np.float32)

DIFOP_ANGLE_TO_RING = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]

# Extrinsic matrix from config_lingang.json
# Site X = System Y, Site Y = System Z, Site Z = 9.4 - System X
EXTRINSIC_MATRIX = np.array([
    [ 0.0,  1.0,  0.0,  0.0 ],
    [ 0.0,  0.0,  1.0,  0.0 ],
    [-1.0,  0.0,  0.0,  9.4 ],
    [ 0.0,  0.0,  0.0,  1.0 ]
], dtype=np.float32)

# ROI (Region of Interest) in Site Coordinates
# 场区空间范围与地面滤除（黄色立体框）：
ROI_CONFIG = {
    "x_min": -32.50,
    "x_max": 21.00,
    "y_min": -3.00,
    "y_max": 4.00,
    "z_min": 0.50,
    "z_max": 6.00,
    "ground_z_threshold": 0.20
}

# 车帮高度识别配置（连续立面自底向上搜索与断档判定算法）
# 注意：车帮测高不应受到 ROI 的 Zmax 截断，默认搜索上限开放至 20.0m
BOARD_HEIGHT_CONFIG = {
    "x_margin": 0.65,             # 粗网格边界有约0.2675m量化偏差，半宽覆盖真实顶部
    "z_min": 0.50,                # 搜索起始高度下限 (0.50m，避开地面反射噪声)
    "z_max": 20.00,               # 搜索上限开放至20m（不受ROI Zmax截断）
    "max_z_gap": 0.35,            # 判定垂直不连续的大断档阈值 (0.35m)
    "min_continuous_pts": 5,      # 有效连续立面段所需的最少点数
    "top_percentile": 98.0        # 连续段顶部的分位数 (98.0%，抑制极值尖峰噪点)
}

# X 轴粗聚类（圈车）默认参数
COARSE_CLUSTER_CONFIG = {
    "y_grid_num": 200,
    "cluster_min_points": 3,
    "cluster_max_gap": 3,
    "cluster_min_span": 6
}

# 师傅方案一：车厢位置检测（直方图切片投票法 + 师傅Z值均匀性与立面限定值）
CABIN_DETECTOR_CONFIG = {
    "bin_size": 0.17,              # 切格大小 (0.17m)
    "min_peak_pts": 25,            # 车帮寻峰门限 (25点)
    "min_cabin_len": 6.00,         # 最小车长限制 (6.00m，完美兼容停稳车厢与长半挂)
    "max_cabin_len": 24.00,        # 最大车长限制 (24.00m)
    "enable_z_uniformity": True,   # 启用师傅限定值
    "min_board_height": 1.80,      # 车帮垂直立面高度门限 (1.80m，秒滤车顶平盖)
    "max_empty_gap": 2.50,         # 允许的最大连续空档 (2.50m)
    "max_base_jump": 1.00          # 底板连续性最大跳变 (1.00m)
}

# 顺序回放的端点跟踪参数；不改变单帧几何识别的默认接口。
CABIN_TRACKING_CONFIG = {
    "position_slack": 0.50,        # 端点位置余量，单位 m
    "max_speed": 5.00,             # 端点最大运动速度，单位 m/s
    "length_tolerance": 0.60,      # 相邻可靠结果的车厢长度差，单位 m
    "reset_gap": 1.00,             # 超过此秒数后重新确认，暂不丢弃身份参考
    "identity_memory": 3.00,       # 身份参考最多保留秒数，超期按新目标捕获
    "recovery_acceleration": 1.00, # 恢复预测的位置余量加速度，单位 m/s²
    "profile_tolerance": 0.50,     # 两端顶部及立面高度跨度允许变化，单位 m
    "confirm_frames": 3           # 初次捕获/重新捕获所需连续候选帧数
}

HEIGHT_TIMESERIES_CONFIG = {
    "median_window": 5,           # 只使用当前帧及历史帧，必须为奇数
    "max_time_gap": 0.30,         # 相邻帧最大时间间隔，单位 s
    "max_height_spread": 0.20,     # 窗口内最大高度跨度，单位 m
    "preserve_short_gaps": True,  # 短缺测保留限时历史，缺测帧仍不输出
    "max_history_age": 0.70,      # 历史样本最长期限，单位 s
    "consistency_threshold": 0.30 # 两端融合门限，单位 m
}

# 卡尔曼滤波开关制与调参配置（优化点云缺失帧小漏检与平滑）
KALMAN_CONFIG = {
    "enabled": True,                   # 默认在统一测高中启用卡尔曼，界面可切回中值
    "max_coast_frames": 5,            # 最大外推补帧数（容忍连续缺失帧数）
    "max_coast_time": 0.60,           # 最大外推保持时间 (s)
    "pos_process_noise_q": 10.0,      # 允许正常起停加速度，减少恒速模型停车过冲
    "pos_measure_noise_r": 0.15,      # 位置测量噪声 R
    "height_process_noise_q": 0.001,  # 高度过程噪声 Q
    "height_measure_noise_r": 0.03,   # 高度测量噪声 R
    "height_init_cov": 0.10,          # 高度初始误差协方差 P0
    "height_gate_limit": 0.25         # 高度残差门限 (m)：防止底板噪声污染
}

# 临港 PLC 接口输出配置 (单位: mm)
# 约定：车辆前进为现场 X 正方向，前车帮为大 X 端 (车头)，后车帮为小 X 端 (车尾)
PLC_CONFIG = {
    "lidar_x_mm": 5750,            # PLC 坐标零点偏移，5.75 m = 5750 mm
    "reg_front_height": 1,         # 寄存器 1 (VW2502): 前车帮(大X端)顶部 Z 高度 (mm)
    "reg_rear_height": 7,          # 寄存器 7 (VW2514): 后车帮(小X端)顶部 Z 高度 (mm)
    "reg_head_position": 43,       # 寄存器 43 (VW2586): 车头(大X端)位置 = 5750 + X_high * 1000 (mm)
    "reg_tail_position": 44,       # 寄存器 44 (VW2588): 车尾(小X端)位置 = 5750 + X_low * 1000 (mm)
    "position_signed": True        # PLC 位置使用有符号 INT16，范围 -32768~32767 mm
}

# Frame cut configuration
DEFAULT_CUT_ANGLE_DEG = 180.0
