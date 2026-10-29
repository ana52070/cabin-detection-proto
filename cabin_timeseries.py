# -*- coding: utf-8 -*-
"""
车帮高度四图联动：顺序端点跟踪、因果历史中值、卡尔曼滤波（开关制及调参）及两端一致性融合。

严格统一端点语义：
- x_low / z_low (小 X 端) 与 x_high / z_high (大 X 端) 保持全局一致无歧义。
- diff: |z_low - z_high|
- z_final: 综合高度（两端一致性 <= 0.30m 时融合）

区分来源与统计：
- reliable_measured: 可靠实测
- local_reacquired: 局部重新识别（通过立面和车帮位置几何门控）
- predicted: 纯卡尔曼外推预测
- invalid: 失效/超时

统一计算通路：提供 run_cabin_timeseries_pipeline，供 GUI、导出 CSV 及验证脚本复用。
"""

import sys
import os
import copy
from collections import deque
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, replace

from PySide6.QtCore import Qt, Signal, QThread, Slot
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QProgressBar, QCheckBox, QFileDialog, QMessageBox, QFrame,
    QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget
)
import pyqtgraph as pg
import pyqtgraph.exporters

from tm16_decoder import TM16Frame
from short_edge_detector import ShortEdgeDetector
from cabin_detector import CabinDetector, CabinPositionResult
from cabin_tracking import CabinPositionTracker
from cabin_measurement import CabinMeasurementCollector, measure_continuous_board_height
from cabin_kalman import KalmanConfig, PositionKalmanFilter, CabinHeightKalmanFilter
import config


@dataclass
class TimeSeriesPoint:
    frame_idx: int
    t_rel: float                                # Relative time in seconds
    z_front_raw: Optional[float] = None        # Low X (front in detector)
    z_rear_raw: Optional[float] = None         # High X (rear in detector)
    pass_front_qual: bool = False
    pass_rear_qual: bool = False
    z_front_med: Optional[float] = None        # Low X filtered/predicted
    z_rear_med: Optional[float] = None         # High X filtered/predicted
    diff_med: Optional[float] = None
    z_final: Optional[float] = None
    front_reason: str = ""
    rear_reason: str = ""

    # 扩展属性：统一命名与来源区分
    x_low: Optional[float] = None
    x_high: Optional[float] = None
    cabin_length: Optional[float] = None
    point_type_low: str = "invalid"             # 'reliable_measured', 'local_reacquired', 'predicted', 'invalid'
    point_type_high: str = "invalid"            # 'reliable_measured', 'local_reacquired', 'predicted', 'invalid'
    time_since_reliable_low: Optional[float] = None
    time_since_reliable_high: Optional[float] = None
    is_gap_fill: bool = False
    position_type: str = 'invalid'
    x_low_raw: Optional[float] = None
    x_high_raw: Optional[float] = None
    cabin_length_raw: Optional[float] = None
    position_reason: str = ''

    @property
    def z_low_raw(self) -> Optional[float]:
        return self.z_front_raw

    @z_low_raw.setter
    def z_low_raw(self, val: Optional[float]):
        self.z_front_raw = val

    @property
    def z_high_raw(self) -> Optional[float]:
        return self.z_rear_raw

    @z_high_raw.setter
    def z_high_raw(self, val: Optional[float]):
        self.z_rear_raw = val

    @property
    def pass_low_qual(self) -> bool:
        return self.pass_front_qual

    @pass_low_qual.setter
    def pass_low_qual(self, val: bool):
        self.pass_front_qual = val

    @property
    def pass_high_qual(self) -> bool:
        return self.pass_rear_qual

    @pass_high_qual.setter
    def pass_high_qual(self, val: bool):
        self.pass_rear_qual = val

    @property
    def z_low_filtered(self) -> Optional[float]:
        return self.z_front_med

    @z_low_filtered.setter
    def z_low_filtered(self, val: Optional[float]):
        self.z_front_med = val

    @property
    def z_high_filtered(self) -> Optional[float]:
        return self.z_rear_med

    @z_high_filtered.setter
    def z_high_filtered(self, val: Optional[float]):
        self.z_rear_med = val

    @property
    def diff_filtered(self) -> Optional[float]:
        return self.diff_med

    @diff_filtered.setter
    def diff_filtered(self, val: Optional[float]):
        self.diff_med = val

    @property
    def low_reason(self) -> str:
        return self.front_reason

    @low_reason.setter
    def low_reason(self, val: str):
        self.front_reason = val

    @property
    def high_reason(self) -> str:
        return self.rear_reason

    @high_reason.setter
    def high_reason(self, val: str):
        self.rear_reason = val


def check_local_board_geometry(roi_pts: np.ndarray, x_target: float,
                               x_tol: float = 0.20, min_pts: int = 8,
                               min_span: float = 0.90, height_config=None,
                               roi=None) -> Optional[float]:
    """
    检查局部重新识别候选点云是否满足真实车帮立面支持：
    1. 候选点数 >= min_pts
    2. 立面垂直跨度 (p98 - p5) >= min_span (防止底板或地面噪点)
    3. 顶部高度在合理区间 [4.0, 7.5]
    """
    if roi_pts is None or len(roi_pts) == 0:
        return None
    h_cfg = height_config if height_config is not None else config.BOARD_HEIGHT_CONFIG
    bounds = roi if roi is not None else config.ROI_CONFIG
    h, _, _, segment = measure_continuous_board_height(
        roi_pts, x_target, **h_cfg,
        y_min=bounds['y_min'], y_max=bounds['y_max'])
    if h is None or len(segment) < min_pts:
        return None
    p5, p98 = np.percentile(segment[:, 2], [5, 98])
    span = p98 - p5
    if span < min_span:
        return None
    if not (4.0 <= p98 <= 7.5):
        return None
    return h


class CabinTimeSeriesAnalyzer:
    """两端独立质量门控、完整奇数历史窗口及融合，支持卡尔曼滤波与因果中值双通路。"""
    def __init__(self,
                 z_min_valid: float = 4.0,
                 z_max_valid: float = 7.5,
                 median_window: int = 5,
                 max_time_gap: float = 0.30,
                 consistency_threshold: float = 0.30,
                 max_height_spread: float = 0.20,
                 preserve_short_gaps: bool = False,
                 max_history_age: float = 0.70,
                 kalman_config: Optional[dict] = None):
        if not isinstance(median_window, int) or median_window < 3 or median_window % 2 == 0:
            raise ValueError("历史中值窗口必须为不小于3的奇数")
        limits = (z_min_valid, z_max_valid, max_time_gap,
                  consistency_threshold, max_height_spread, max_history_age)
        if not all(np.isfinite(v) for v in limits):
            raise ValueError("滤波参数必须有限")
        if z_min_valid >= z_max_valid or min(limits[2:]) <= 0:
            raise ValueError("滤波范围或门限无效")
        self.z_min_valid = z_min_valid
        self.z_max_valid = z_max_valid
        self.median_window = median_window
        self.max_time_gap = max_time_gap
        self.consistency_threshold = consistency_threshold
        self.max_height_spread = max_height_spread
        self.preserve_short_gaps = preserve_short_gaps
        self.max_history_age = max_history_age
        self.kalman_config = copy.deepcopy(kalman_config) if kalman_config is not None else None

    def analyze_frames(self, frames_data: List[dict]) -> Dict:
        """逐帧滤波分析。支持外部传入统一 frames_data 字典列表。"""
        use_kalman = self.kalman_config and self.kalman_config.get('enabled', False)
        if use_kalman:
            return self._analyze_frames_kalman(frames_data)
        else:
            return self._analyze_frames_median(frames_data)

    def _analyze_frames_kalman(self, frames_data: List[dict]) -> Dict:
        kf_cfg = KalmanConfig(**self.kalman_config)
        height_kf = CabinHeightKalmanFilter(
            kf_cfg,
            z_min=self.z_min_valid,
            z_max=self.z_max_valid,
            consistency_threshold=self.consistency_threshold
        )
        pts: List[TimeSeriesPoint] = []
        counts = {
            'raw_low_count': 0, 'raw_high_count': 0,
            'pass_low_count': 0, 'pass_high_count': 0,
            'local_reacquired_low_count': 0, 'local_reacquired_high_count': 0,
            'predicted_low_count': 0, 'predicted_high_count': 0,
            'invalid_low_count': 0, 'invalid_high_count': 0,
            'out_low_count': 0, 'out_high_count': 0,
            'final_count': 0
        }

        previous_time = None
        previous_epoch = None
        allowed_max_gap = kf_cfg.max_coast_time

        for fd in frames_data:
            t, idx = fd['t_rel'], fd['frame_idx']
            if not np.isfinite(t):
                raise ValueError("帧时间必须有限")

            epoch = fd.get('track_epoch', 0)
            broken = (previous_time is not None and
                      (t <= previous_time or t - previous_time > allowed_max_gap
                       or epoch != previous_epoch))

            if broken:
                height_kf.reset()

            raw_l = fd.get('z_low_raw') if 'z_low_raw' in fd else fd.get('z_front_raw')
            raw_h = fd.get('z_high_raw') if 'z_high_raw' in fd else fd.get('z_rear_raw')

            # 统计原始候选测量
            if raw_l is not None and np.isfinite(raw_l):
                counts['raw_low_count'] += 1
            if raw_h is not None and np.isfinite(raw_h):
                counts['raw_high_count'] += 1

            # 物理有效性初筛
            is_valid_cand_l = (raw_l is not None and np.isfinite(raw_l) and self.z_min_valid <= raw_l <= self.z_max_valid)
            is_valid_cand_h = (raw_h is not None and np.isfinite(raw_h) and self.z_min_valid <= raw_h <= self.z_max_valid)

            zl_in = raw_l if is_valid_cand_l else None
            zh_in = raw_h if is_valid_cand_h else None

            zl, zh, diff, z_final, info_l, info_h = height_kf.process_detailed(zl_in, zh_in, t)

            # 严格区分来源
            src_l = info_l.get("source", "invalid")
            src_h = info_h.get("source", "invalid")
            is_local_l = fd.get("is_local_reacquired_low", False)
            is_local_h = fd.get("is_local_reacquired_high", False)

            # 实测质量通过数（决不混入纯预测，决不超过原始候选数）
            pass_low_qual = bool(is_valid_cand_l and src_l in ("reliable_measured", "step_confirmed"))
            pass_high_qual = bool(is_valid_cand_h and src_h in ("reliable_measured", "step_confirmed"))

            if pass_low_qual:
                counts['pass_low_count'] += 1
            if pass_high_qual:
                counts['pass_high_count'] += 1

            type_l = "invalid"
            if zl is not None:
                if is_local_l and pass_low_qual:
                    type_l = "local_reacquired"
                    counts['local_reacquired_low_count'] += 1
                elif src_l in ("reliable_measured", "step_confirmed"):
                    type_l = "reliable_measured"
                else:
                    type_l = "predicted"
                    counts['predicted_low_count'] += 1
                counts['out_low_count'] += 1
            else:
                counts['invalid_low_count'] += 1

            type_h = "invalid"
            if zh is not None:
                if is_local_h and pass_high_qual:
                    type_h = "local_reacquired"
                    counts['local_reacquired_high_count'] += 1
                elif src_h in ("reliable_measured", "step_confirmed"):
                    type_h = "reliable_measured"
                else:
                    type_h = "predicted"
                    counts['predicted_high_count'] += 1
                counts['out_high_count'] += 1
            else:
                counts['invalid_high_count'] += 1

            if z_final is not None:
                counts['final_count'] += 1

            p = TimeSeriesPoint(
                frame_idx=idx,
                t_rel=t,
                z_front_raw=raw_l,
                z_rear_raw=raw_h,
                pass_front_qual=pass_low_qual,
                pass_rear_qual=pass_high_qual,
                z_front_med=zl,
                z_rear_med=zh,
                diff_med=diff,
                z_final=z_final,
                front_reason=info_l.get("reason", ""),
                rear_reason=info_h.get("reason", ""),
                x_low=fd.get('x_low'),
                x_high=fd.get('x_high'),
                cabin_length=fd.get('cabin_length'),
                point_type_low=type_l,
                point_type_high=type_h,
                time_since_reliable_low=info_l.get("time_since_reliable"),
                time_since_reliable_high=info_h.get("time_since_reliable"),
                is_gap_fill=fd.get('is_gap_fill', False),
                position_type=fd.get('position_type', 'invalid'),
                x_low_raw=fd.get('x_low_raw'), x_high_raw=fd.get('x_high_raw'),
                cabin_length_raw=fd.get('cabin_length_raw'), position_reason=fd.get('position_reason', '')
            )

            pts.append(p)
            previous_time = t
            previous_epoch = epoch

        res = dict(points=pts, consistency_threshold=self.consistency_threshold, **counts)
        # 兼容旧字段统计
        res['raw_front_count'] = counts['raw_low_count']
        res['raw_rear_count'] = counts['raw_high_count']
        res['pass_front_count'] = counts['pass_low_count']
        res['pass_rear_count'] = counts['pass_high_count']
        res['med_front_count'] = counts['out_low_count']
        res['med_rear_count'] = counts['out_high_count']
        return res

    def _analyze_frames_median(self, frames_data: List[dict]) -> Dict:
        """完整保留因果中值逻辑与预热检查，与基线100%严格一致。"""
        pts = []
        windows = {side: deque(maxlen=self.median_window)
                   for side in ('front', 'rear')}
        times = {side: deque(maxlen=self.median_window) for side in windows}
        pending = {side: deque(maxlen=self.median_window) for side in windows}
        previous_time = None
        previous_idx = None
        previous_epoch = None
        counts = {name: 0 for name in (
            'raw_front_count', 'raw_rear_count', 'pass_front_count',
            'pass_rear_count', 'med_front_count', 'med_rear_count', 'final_count')}
        for fd in frames_data:
            t, idx = fd['t_rel'], fd['frame_idx']
            if not np.isfinite(t):
                raise ValueError("帧时间必须有限")
            broken = (previous_time is not None and
                      (t <= previous_time or t - previous_time > self.max_time_gap
                       or idx != previous_idx + 1
                       or fd.get('track_epoch', 0) != previous_epoch))
            p = TimeSeriesPoint(frame_idx=idx, t_rel=t,
                                z_front_raw=fd.get('z_low_raw', fd.get('z_front_raw')),
                                z_rear_raw=fd.get('z_high_raw', fd.get('z_rear_raw')),
                                x_low=fd.get('x_low'),
                                x_high=fd.get('x_high'),
                                cabin_length=fd.get('cabin_length'),
                                position_type=fd.get('position_type', 'invalid'),
                                x_low_raw=fd.get('x_low_raw'), x_high_raw=fd.get('x_high_raw'),
                                cabin_length_raw=fd.get('cabin_length_raw'), position_reason=fd.get('position_reason', ''))
            seeds = fd.get('warmup_samples', [])
            if seeds:
                if fd.get('status') != 'ok' or fd.get('track_event') not in ('resumed', 'reacquired'):
                    raise ValueError("只有本帧确认的候选可用于历史预热")
                if any(not np.isfinite(s['t_rel']) or s['t_rel'] >= t or s['frame_idx'] >= idx
                       for s in seeds):
                    raise ValueError("预热样本必须早于当前帧，不允许未来数据")
                if any(b['frame_idx'] != a['frame_idx'] + 1 or b['t_rel'] <= a['t_rel']
                       for a, b in zip(seeds, seeds[1:] + [fd])):
                    raise ValueError("预热候选必须按帧连续且时间递增")
            for side in ('front', 'rear'):
                window = windows[side]
                sample_times = times[side]
                candidate = pending[side]
                if broken:
                    window.clear()
                    sample_times.clear()
                    candidate.clear()
                if seeds:
                    window.clear()
                    sample_times.clear()
                    candidate.clear()
                    previous_seed_time = None
                    for seed in seeds:
                        key = 'z_low_raw' if side == 'front' else 'z_high_raw'
                        value = seed.get(key, seed.get(f'z_{side}_raw'))
                        age = t - seed['t_rel']
                        if (value is None or not np.isfinite(value)
                                or not self.z_min_valid <= value <= self.z_max_valid
                                or age > self.max_history_age):
                            window.clear()
                            sample_times.clear()
                            previous_seed_time = None
                            continue
                        if (previous_seed_time is not None
                                and seed['t_rel'] - previous_seed_time > self.max_time_gap):
                            window.clear()
                            sample_times.clear()
                        if window and max(max(window), value) - min(min(window), value) > self.max_height_spread:
                            window.clear()
                            sample_times.clear()
                        window.append(float(value))
                        sample_times.append(seed['t_rel'])
                        previous_seed_time = seed['t_rel']
                    if sample_times and t - sample_times[-1] > self.max_time_gap:
                        window.clear()
                        sample_times.clear()
                while sample_times and t - sample_times[0] > self.max_history_age:
                    sample_times.popleft()
                    window.popleft()
                z = getattr(p, f'z_{side}_raw')
                if z is not None:
                    counts[f'raw_{side}_count'] += 1
                valid = (fd.get('status', 'ok') == 'ok' and z is not None
                         and np.isfinite(z) and self.z_min_valid <= z <= self.z_max_valid)
                setattr(p, f'pass_{side}_qual', bool(valid))
                reason = '无效或缺测'
                point_type = 'invalid'
                if not valid:
                    candidate.clear()
                    if not self.preserve_short_gaps:
                        window.clear()
                        sample_times.clear()
                else:
                    counts[f'pass_{side}_count'] += 1
                    if len(window) == self.median_window:
                        window.popleft()
                        sample_times.popleft()
                    if window and max(max(window), z) - min(min(window), z) > self.max_height_spread:
                        if self.preserve_short_gaps:
                            if candidate and (t - candidate[0][0] > self.max_history_age
                                    or max(max(v for _, v in candidate), z)
                                    - min(min(v for _, v in candidate), z) > self.max_height_spread):
                                candidate.clear()
                            candidate.append((t, float(z)))
                            if len(candidate) < self.median_window:
                                setattr(p, f'{side}_reason', '高度突变，候选待确认')
                                continue
                            window.clear()
                            sample_times.clear()
                            for sample_t, value in list(candidate)[:-1]:
                                window.append(value)
                                sample_times.append(sample_t)
                        else:
                            window.clear()
                            sample_times.clear()
                        reason = '高度突变，重新预热'
                    else:
                        reason = '断档后预热' if broken else '连续样本预热'
                    candidate.clear()
                    window.append(float(z))
                    sample_times.append(t)
                    if len(window) == self.median_window:
                        setattr(p, f'z_{side}_med', float(np.median(window)))
                        counts[f'med_{side}_count'] += 1
                        reason = '有效'
                        point_type = 'reliable_measured'
                setattr(p, f'{side}_reason', reason)
                setattr(p, f'point_type_{"low" if side=="front" else "high"}', point_type)
            if p.z_front_med is not None and p.z_rear_med is not None:
                p.diff_med = abs(p.z_front_med - p.z_rear_med)
                if p.diff_med <= self.consistency_threshold:
                    p.z_final = 0.5 * (p.z_front_med + p.z_rear_med)
                    counts['final_count'] += 1
            pts.append(p)
            previous_time, previous_idx = t, idx
            previous_epoch = fd.get('track_epoch', 0)
        res = dict(points=pts, consistency_threshold=self.consistency_threshold, **counts)
        res['raw_low_count'] = counts['raw_front_count']
        res['raw_high_count'] = counts['raw_rear_count']
        res['pass_low_count'] = counts['pass_front_count']
        res['pass_high_count'] = counts['pass_rear_count']
        res['out_low_count'] = counts['med_front_count']
        res['out_high_count'] = counts['med_rear_count']
        res['local_reacquired_low_count'] = 0
        res['local_reacquired_high_count'] = 0
        res['predicted_low_count'] = 0
        res['predicted_high_count'] = 0
        res['invalid_low_count'] = len(frames_data) - counts['med_front_count']
        res['invalid_high_count'] = len(frames_data) - counts['med_rear_count']
        return res


def run_cabin_timeseries_pipeline(
    frames: List[TM16Frame],
    roi_detector: ShortEdgeDetector,
    cabin_detector: CabinDetector,
    tracking_config: Optional[dict] = None,
    timeseries_config: Optional[dict] = None,
    kalman_config: Optional[dict] = None,
    progress_callback: Optional[Any] = None,
    cancel_check: Optional[Any] = None,
    height_config: Optional[dict] = None
) -> Dict[str, Any]:
    """
    统一计算通路核心流程：
    供 GUI 线程、CSV 导出、绘图及验证脚本共用。
    """
    total = len(frames)
    if total == 0:
        return {'points': [], 'positions': [], 'frames_data': []}

    trk_cfg = tracking_config if tracking_config is not None else config.CABIN_TRACKING_CONFIG
    ts_cfg = timeseries_config if timeseries_config is not None else config.HEIGHT_TIMESERIES_CONFIG
    k_cfg = kalman_config

    t0 = frames[0].timestamp
    frames_data: List[dict] = []
    positions: List[CabinPositionResult] = []

    cabin_detector.roi = dict(roi_detector.roi)
    cabin_detector.coarse_detector = copy.deepcopy(roi_detector)
    tracker = CabinPositionTracker(cabin_detector, **trk_cfg)
    h_cfg = dict(height_config if height_config is not None else config.BOARD_HEIGHT_CONFIG)
    collector = CabinMeasurementCollector(tracker.confirm_frames, roi=roi_detector.roi, height_config=h_cfg)

    use_kalman = bool(k_cfg and k_cfg.get('enabled', False))
    pos_kf = PositionKalmanFilter(KalmanConfig(**k_cfg)) if use_kalman else None

    last_processed_time: Optional[float] = None
    nominal_dt = 0.10 # TM16 10Hz nominal interval
    packet_history = deque(maxlen=31)
    last_position_epoch = None

    for idx, f in enumerate(frames):
        if cancel_check and cancel_check():
            return {}

        t_now = f.timestamp
        t_rel = t_now - t0
        pts_site = f.points_site
        roi_pts = roi_detector.filter_roi(pts_site)

        # 检查帧间时间差，若存在缺测空档且卡尔曼开启，则在空档内以 nominal_dt 推进前向预测
        if use_kalman and pos_kf and pos_kf.is_initialized and last_processed_time is not None:
            gap_dt = t_now - last_processed_time
            if gap_dt > nominal_dt * 1.5:
                # 存在时间空档：按固定周期产生前向预测，直至到达上限或到达当前帧时间
                t_step = last_processed_time + nominal_dt
                while t_step < t_now - 1e-4:
                    coasted_step = pos_kf.coast(t_step)
                    if coasted_step is None:
                        # 超过保持时限或步数，外推自然终止失效
                        break
                    xl_s, xh_s, len_s = coasted_step
                    t_step_rel = t_step - t0
                    gap_row = dict(
                        frame_idx=idx - 1, # 关联至最近的前序点云帧供三维显示
                        t_rel=t_step_rel,
                        status='predicted_coasting',
                        x_low=xl_s, x_high=xh_s,
                        x_front=xh_s, x_rear=xl_s,
                        cabin_length=len_s,
                        track_epoch=tracker.epoch,
                        track_event='kalman_predict_gap',
                        candidate_id=tracker.candidate_id,
                        z_low_raw=None, z_high_raw=None,
                        z_front_raw=None, z_rear_raw=None,
                        is_local_reacquired_low=False,
                        is_local_reacquired_high=False,
                        is_gap_fill=True,
                        position_type='predicted',
                        warmup_samples=[]
                    )
                    frames_data.append(gap_row)
                    t_step += nominal_dt

        # 运行主检测与顺序跟踪
        packet_reference = float(np.median(packet_history)) if packet_history else f.packet_count
        frame_complete = f.packet_count > 0 and f.packet_count >= .9*packet_reference
        if frame_complete:
            packet_history.append(f.packet_count)
        res = tracker.update(roi_pts, t_now, frame_complete=frame_complete)
        display_res = res

        if use_kalman and pos_kf:
            if res.status == 'ok':
                if last_position_epoch != res.track_epoch:
                    pos_kf.reset()
                last_position_epoch = res.track_epoch
                xl_meas = min(res.x_front, res.x_rear)
                xh_meas = max(res.x_front, res.x_rear)
                pos_kf.update(x_low=xl_meas, x_high=xh_meas, timestamp=t_now)
                xl_out, xh_out, len_out = pos_kf.current_estimate()
                display_res = replace(res, x_front=xh_out, x_rear=xl_out, cabin_length=len_out,
                                      cabin_box=(xl_out, xh_out, *res.cabin_box[2:]) if res.cabin_box else None)
                collected = collector.collect(idx, t_rel, pts_site, res, tracker)
                raw_f = collected.get('z_front_raw')
                raw_r = collected.get('z_rear_raw')
                row = dict(
                    frame_idx=idx, t_rel=t_rel, status='ok',
                    x_low=xl_out, x_high=xh_out,
                    x_front=xh_out, x_rear=xl_out,
                    cabin_length=len_out, track_epoch=res.track_epoch,
                    track_event=res.track_event, candidate_id=tracker.candidate_id,
                    z_low_raw=raw_r, z_high_raw=raw_f,
                    z_front_raw=raw_f, z_rear_raw=raw_r,
                    is_local_reacquired_low=res.track_event == 'local_reacquired' and raw_r is not None,
                    is_local_reacquired_high=res.track_event == 'local_reacquired' and raw_f is not None,
                    is_gap_fill=False,
                    warmup_samples=collected.get('warmup_samples', [])
                )
                frames_data.append(row)
            else:
                coasted = pos_kf.coast(t_now)
                if coasted is not None:
                    xl_c, xh_c, len_c = coasted
                    display_res = replace(res, status='predicted', x_front=xh_c, x_rear=xl_c,
                                          cabin_length=len_c, track_event='kalman_predict',
                                          cabin_box=(xl_c, xh_c, roi_detector.roi['y_min'], roi_detector.roi['y_max'],
                                                     roi_detector.roi['z_min'], roi_detector.roi['z_max']))
                    # 几何门控的局部重新识别
                    # 使用完整高度范围，但遵守当前ROI的横向边界。
                    height_pts = pts_site[(pts_site[:, 1] >= roi_detector.roi['y_min']) &
                                          (pts_site[:, 1] <= roi_detector.roi['y_max'])]
                    zl_local = check_local_board_geometry(height_pts, xl_c, height_config=h_cfg, roi=roi_detector.roi)
                    zh_local = check_local_board_geometry(height_pts, xh_c, height_config=h_cfg, roi=roi_detector.roi)

                    row = dict(
                        frame_idx=idx, t_rel=t_rel,
                        status='ok' if (zl_local is not None or zh_local is not None) else 'kalman_predict',
                        x_low=xl_c, x_high=xh_c,
                        x_front=xh_c, x_rear=xl_c,
                        cabin_length=len_c, track_epoch=tracker.epoch,
                        track_event='kalman_reacquire' if (zl_local is not None or zh_local is not None) else 'kalman_predict',
                        candidate_id=tracker.candidate_id,
                        z_low_raw=zl_local, z_high_raw=zh_local,
                        z_front_raw=zh_local, z_rear_raw=zl_local,
                        is_local_reacquired_low=bool(zl_local is not None),
                        is_local_reacquired_high=bool(zh_local is not None),
                        is_gap_fill=False,
                        warmup_samples=[]
                    )
                    frames_data.append(row)
                else:
                    collected = collector.collect(idx, t_rel, pts_site, res, tracker)
                    raw_f = collected.get('z_front_raw')
                    raw_r = collected.get('z_rear_raw')
                    xl_m = min(res.x_front, res.x_rear) if (res.x_front is not None and res.x_rear is not None) else None
                    xh_m = max(res.x_front, res.x_rear) if (res.x_front is not None and res.x_rear is not None) else None
                    row = dict(
                        frame_idx=idx, t_rel=t_rel, status=res.status,
                        x_low=xl_m, x_high=xh_m,
                        x_front=res.x_front, x_rear=res.x_rear,
                        cabin_length=res.cabin_length, track_epoch=res.track_epoch,
                        track_event=res.track_event, candidate_id=tracker.candidate_id,
                        z_low_raw=raw_r,
                        z_high_raw=raw_f,
                        z_front_raw=raw_f,
                        z_rear_raw=raw_r,
                        is_local_reacquired_low=False,
                        is_local_reacquired_high=False,
                        is_gap_fill=False,
                        warmup_samples=collected.get('warmup_samples', [])
                    )
                    frames_data.append(row)
        else:
            collected = collector.collect(idx, t_rel, pts_site, res, tracker)
            raw_f = collected.get('z_front_raw')
            raw_r = collected.get('z_rear_raw')
            xl_m = min(res.x_front, res.x_rear) if (res.x_front is not None and res.x_rear is not None) else None
            xh_m = max(res.x_front, res.x_rear) if (res.x_front is not None and res.x_rear is not None) else None
            row = dict(
                frame_idx=idx, t_rel=t_rel, status=res.status,
                x_low=xl_m, x_high=xh_m,
                x_front=res.x_front, x_rear=res.x_rear,
                cabin_length=res.cabin_length, track_epoch=res.track_epoch,
                track_event=res.track_event, candidate_id=tracker.candidate_id,
                z_low_raw=raw_r,
                z_high_raw=raw_f,
                z_front_raw=raw_f,
                z_rear_raw=raw_r,
                is_local_reacquired_low=False,
                is_local_reacquired_high=False,
                is_gap_fill=False,
                warmup_samples=collected.get('warmup_samples', [])
            )
            frames_data.append(row)

        # 历史front/rear在检测器中指大/小X；预热也必须转换到统一端点。
        row = frames_data[-1]
        row['position_type'] = ('local_reacquired' if res.track_event == 'local_reacquired' else
                                'reliable_measured') if res.status == 'ok' else (
            'predicted' if display_res.status == 'predicted' else 'invalid')
        raw = tracker.raw_result
        row.update(x_low_raw=raw.x_rear, x_high_raw=raw.x_front,
                   cabin_length_raw=raw.cabin_length if raw.x_front is not None else None,
                   packet_count=f.packet_count, packet_reference=packet_reference,
                   frame_complete=bool(frame_complete), position_reason=(
                       '位置预测；'+res.z_check_status if display_res.status == 'predicted' else res.z_check_status))
        positions.append(display_res)
        for seed in row.get('warmup_samples', []):
            seed['z_low_raw'] = seed.get('z_rear_raw')
            seed['z_high_raw'] = seed.get('z_front_raw')
        last_processed_time = t_now

        if progress_callback and (idx % 30 == 0 or idx == total - 1):
            progress_callback(idx + 1, total)

    # 运行统一时序分析器
    analyzer = CabinTimeSeriesAnalyzer(**ts_cfg, kalman_config=k_cfg)
    result = analyzer.analyze_frames(frames_data)
    result['positions'] = positions
    result['frames_data'] = frames_data
    return result


class TimeSeriesComputeWorker(QThread):
    """Computes detections and time-series filtering across all frames in background using unified pipeline."""
    progress = Signal(int, int) # current, total
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self,
                 frames: List[TM16Frame],
                 roi_detector: ShortEdgeDetector,
                 cabin_detector: CabinDetector,
                 analyzer: CabinTimeSeriesAnalyzer, height_config=None):
        super().__init__()
        self.frames = frames
        self.roi_detector = copy.deepcopy(roi_detector)
        self.cabin_detector = copy.deepcopy(cabin_detector)
        self.analyzer = copy.deepcopy(analyzer)
        self.height_config = copy.deepcopy(height_config if height_config is not None else config.BOARD_HEIGHT_CONFIG)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if not self.frames:
                self.finished.emit({})
                return

            def on_progress(cur, tot):
                self.progress.emit(cur, tot)

            def check_cancel():
                return self._is_cancelled

            ts_cfg = {
                "z_min_valid": self.analyzer.z_min_valid,
                "z_max_valid": self.analyzer.z_max_valid,
                "median_window": self.analyzer.median_window,
                "max_time_gap": self.analyzer.max_time_gap,
                "consistency_threshold": self.analyzer.consistency_threshold,
                "max_height_spread": self.analyzer.max_height_spread,
                "preserve_short_gaps": self.analyzer.preserve_short_gaps,
                "max_history_age": self.analyzer.max_history_age,
            }

            result = run_cabin_timeseries_pipeline(
                frames=self.frames,
                roi_detector=self.roi_detector,
                cabin_detector=self.cabin_detector,
                tracking_config=config.CABIN_TRACKING_CONFIG,
                timeseries_config=ts_cfg,
                kalman_config=self.analyzer.kalman_config,
                progress_callback=on_progress,
                cancel_check=check_cancel,
                height_config=self.height_config
            )

            if not self._is_cancelled:
                self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class CabinTimeSeriesWindow(QMainWindow):
    """
    Dedicated 4-Subplot Real-time Time-Series Monitor Window with Linked Scrubbing.
    Unified naming: x_low / x_high, z_low / z_high.
    Clear distinction between measured/local_reacquired and predicted points.
    """
    frame_selected = Signal(int) # Emitted when user clicks/drags in time-series plots
    results_ready = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("车帮高度与位置时序（高度四图 / 前后位置与车长三图）")
        self.resize(1260, 920)

        self.analyzer = CabinTimeSeriesAnalyzer(**config.HEIGHT_TIMESERIES_CONFIG,
                                                kalman_config=getattr(config, 'KALMAN_CONFIG', None))
        self.worker: Optional[TimeSeriesComputeWorker] = None
        self.points_data: List[TimeSeriesPoint] = []
        self.t_rel_list: List[float] = []
        self.position_results = []

        self._init_ui()
        self._apply_dark_theme()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # Top Control Bar
        top_bar = QHBoxLayout()
        self.lbl_title = QLabel("📊 空车完整回放: 车帮顶部 Z 时序曲线 (实测/局部恢复/外推严格区分)")
        self.lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #66b2ff;")
        top_bar.addWidget(self.lbl_title)

        top_bar.addStretch()

        self.lbl_progress = QLabel("")
        self.lbl_progress.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        top_bar.addWidget(self.lbl_progress)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(140)
        self.progress_bar.setVisible(False)
        top_bar.addWidget(self.progress_bar)

        self.btn_recompute = QPushButton("🔄 重新提取与计算")
        self.btn_recompute.setToolTip("以当前 ROI、算法与卡尔曼配置重新解算所有帧的时序曲线")
        top_bar.addWidget(self.btn_recompute)

        self.btn_export = QPushButton("💾 导出高清图")
        self.btn_export.clicked.connect(self._export_image)
        top_bar.addWidget(self.btn_export)

        vbox.addLayout(top_bar)

        # Clean integrated controls (Kalman toggle is embedded directly in top_bar)
        self.chk_kalman = QCheckBox("启用卡尔曼滤波与断检外推")
        k_en = self.analyzer.kalman_config.get('enabled', True) if self.analyzer.kalman_config else True
        self.chk_kalman.setChecked(k_en)
        self.chk_kalman.setStyleSheet("color: #10b981; font-weight: bold;")
        self.chk_kalman.toggled.connect(self._on_kalman_toggled)
        top_bar.insertWidget(2, self.chk_kalman)

        # 保留底层调优参数实例以兼容回调逻辑，默认采用推荐工程参数
        self.sp_max_coast = QSpinBox()
        self.sp_max_coast.setValue(self.analyzer.kalman_config.get('max_coast_frames', 5) if self.analyzer.kalman_config else 5)
        self.sp_coast_time = QDoubleSpinBox()
        self.sp_coast_time.setValue(self.analyzer.kalman_config.get('max_coast_time', 0.60) if self.analyzer.kalman_config else 0.60)
        self.sp_gate_limit = QDoubleSpinBox()
        self.sp_gate_limit.setValue(self.analyzer.kalman_config.get('height_gate_limit', 0.25) if self.analyzer.kalman_config else 0.25)
        self.sp_proc_noise = QDoubleSpinBox()
        self.sp_proc_noise.setDecimals(7)
        self.sp_proc_noise.setRange(0.0000001, 1.0)
        self.sp_proc_noise.setValue(self.analyzer.kalman_config.get('height_process_noise_q', 0.001) if self.analyzer.kalman_config else 0.001)
        self.sp_meas_noise = QDoubleSpinBox()
        self.sp_meas_noise.setDecimals(5)
        self.sp_meas_noise.setRange(0.00001, 1.0)
        self.sp_meas_noise.setValue(self.analyzer.kalman_config.get('height_measure_noise_r', 0.03) if self.analyzer.kalman_config else 0.03)
        # GraphicsLayoutWidget containing 4 subplots
        pg.setConfigOptions(antialias=True)
        self.gl_widget = pg.GraphicsLayoutWidget()
        self.gl_widget.setBackground('#18181f')
        self.tabs = QTabWidget()
        self.tabs.addTab(self.gl_widget, '车帮顶部 Z（四图）')
        self.position_widget = pg.GraphicsLayoutWidget()
        self.position_widget.setBackground('#18181f')
        self.tabs.addTab(self.position_widget, '前后车帮位置与车长（三图）')
        vbox.addWidget(self.tabs, 1)

        # Subplot 1: Low End Top Z (Small X)
        self.p1 = self.gl_widget.addPlot(row=0, col=0)
        self.p1.setLabel('left', '顶部 Z', units='m')
        self.p1.showGrid(x=True, y=True, alpha=0.25)
        self.p1.setYRange(4.0, 7.5, padding=0.05)
        self.p1.setTitle("小X端顶部 Z | 原始 -- 帧 → 质量通过 -- 帧 → 输出 -- 帧", color='#66b2ff', size='11pt')
        self.curve_low_meas = pg.PlotDataItem(
            pen=pg.mkPen('#3b82f6', width=1.5),
            symbol='o', symbolSize=4, symbolBrush='#3b82f6', symbolPen=None, connect='finite'
        )
        self.curve_low_pred = pg.PlotDataItem(
            pen=pg.mkPen('#93c5fd', width=1.5, style=Qt.DashLine),
            symbol='s', symbolSize=3, symbolBrush='#93c5fd', symbolPen=None, connect='finite'
        )
        self.p1.addItem(self.curve_low_meas)
        self.p1.addItem(self.curve_low_pred)

        # Subplot 2: High End Top Z (Large X)
        self.p2 = self.gl_widget.addPlot(row=1, col=0)
        self.p2.setLabel('left', '顶部 Z', units='m')
        self.p2.showGrid(x=True, y=True, alpha=0.25)
        self.p2.setYRange(4.0, 7.5, padding=0.05)
        self.p2.setTitle("大X端顶部 Z | 原始 -- 帧 → 质量通过 -- 帧 → 输出 -- 帧", color='#f97316', size='11pt')
        self.p2.setXLink(self.p1)
        self.curve_high_meas = pg.PlotDataItem(
            pen=pg.mkPen('#f97316', width=1.5),
            symbol='o', symbolSize=4, symbolBrush='#f97316', symbolPen=None, connect='finite'
        )
        self.curve_high_pred = pg.PlotDataItem(
            pen=pg.mkPen('#fdba74', width=1.5, style=Qt.DashLine),
            symbol='s', symbolSize=3, symbolBrush='#fdba74', symbolPen=None, connect='finite'
        )
        self.p2.addItem(self.curve_high_meas)
        self.p2.addItem(self.curve_high_pred)

        # Subplot 3: Absolute Difference
        self.p3 = self.gl_widget.addPlot(row=2, col=0)
        self.p3.setLabel('left', '绝对差', units='m')
        self.p3.showGrid(x=True, y=True, alpha=0.25)
        self.p3.setYRange(0.0, 0.7, padding=0.05)
        self.p3.setTitle("两端顶部绝对差 (仅两端同时有输出时计算)", color='#c084fc', size='11pt')
        self.p3.setXLink(self.p1)

        self.line_thresh = pg.InfiniteLine(
            pos=0.30, angle=0, pen=pg.mkPen('#ef4444', width=1.5, style=Qt.DashLine),
            label='最终高度一致性门限 0.30m',
            labelOpts={'color': '#ef4444', 'position': 0.85, 'anchors': [(0, 0), (0, 0)]}
        )
        self.p3.addItem(self.line_thresh)
        self.curve_diff = pg.PlotDataItem(
            pen=pg.mkPen('#a855f7', width=1.5),
            symbol='o', symbolSize=4, symbolBrush='#a855f7', symbolPen=None, connect='finite'
        )
        self.p3.addItem(self.curve_diff)

        # Subplot 4: Final Fused Top Z
        self.p4 = self.gl_widget.addPlot(row=3, col=0)
        self.p4.setLabel('left', '最终顶部 Z', units='m')
        self.p4.setLabel('bottom', '相对时间 (s) | 实线=实测校正输出，虚线=先验预测输出')
        self.p4.showGrid(x=True, y=True, alpha=0.25)
        self.p4.setYRange(4.0, 7.5, padding=0.05)
        self.p4.setTitle("最终车帮综合高度 (两端一致性 <= 0.30m 融合)", color='#10b981', size='11pt')
        self.p4.setXLink(self.p1)
        self.curve_final_meas = pg.PlotDataItem(
            pen=pg.mkPen('#10b981', width=2.0),
            symbol='o', symbolSize=4, symbolBrush='#10b981', symbolPen=None, connect='finite'
        )
        self.curve_final_pred = pg.PlotDataItem(
            pen=pg.mkPen('#6ee7b7', width=1.8, style=Qt.DashLine),
            symbol='s', symbolSize=3, symbolBrush='#6ee7b7', symbolPen=None, connect='finite'
        )
        self.p4.addItem(self.curve_final_meas)
        self.p4.addItem(self.curve_final_pred)

        self.plots = [self.p1, self.p2, self.p3, self.p4]
        self.position_plots = []
        self.position_curves = []
        self.position_pred_curves = []
        for i, (title, color) in enumerate((('前车帮位置（大 X，质量门控后）', '#f97316'),
                                            ('后车帮位置（小 X，质量门控后）', '#3b82f6'),
                                            ('车长（可靠端点跨度）', '#10b981'))):
            plot = self.position_widget.addPlot(row=i, col=0, title=title)
            plot.setLabel('left', '位置' if i < 2 else '长度', units='m')
            plot.showGrid(x=True, y=True, alpha=.25)
            plot.setXLink(self.p1)
            plot.enableAutoRange(axis='y', enable=True)
            plot.setAutoVisible(y=True)
            self.position_plots.append(plot)
            self.position_curves.append(plot.plot(pen=pg.mkPen(color, width=1.5), connect='finite'))
            self.position_pred_curves.append(plot.plot(pen=pg.mkPen(color, width=1.5, style=Qt.DashLine),
                                                       symbol='s', symbolSize=5, symbolBrush=color,
                                                       symbolPen=None, connect='finite'))
        self.position_plots[-1].setLabel('bottom', '相对时间 (s) | 实线=检测位置，虚线=预测位置')
        self.plots += self.position_plots
        self.cursor_lines = []

        for p in self.plots:
            line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#fbbf24', width=1.5, style=Qt.DashLine), movable=False)
            line.setVisible(False)
            p.addItem(line)
            self.cursor_lines.append(line)
        for scene in {p.scene() for p in self.plots}:
            scene.sigMouseClicked.connect(self._on_plot_clicked)

        # Status readout bar
        bottom_bar = QHBoxLayout()
        self.lbl_status = QLabel("就绪: 点击 [🔄 重新提取与计算] 开始解算")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
        bottom_bar.addWidget(self.lbl_status, 1)

        self.lbl_cursor_info = QLabel("光标位置: --")
        self.lbl_cursor_info.setWordWrap(True)
        self.lbl_cursor_info.setMaximumWidth(850)
        self.lbl_cursor_info.setStyleSheet("color: #fbbf24; font-weight: bold; font-size: 12px;")
        bottom_bar.addWidget(self.lbl_cursor_info)

        vbox.addLayout(bottom_bar)

    def _on_kalman_toggled(self, checked: bool):
        if not self.analyzer.kalman_config:
            self.analyzer.kalman_config = copy.deepcopy(config.KALMAN_CONFIG)
        self.analyzer.kalman_config['enabled'] = checked
        config.KALMAN_CONFIG['enabled'] = checked
        self.invalidate_results()
        self.lbl_status.setText(f"卡尔曼滤波开关已{'启用' if checked else '关闭'}，请点击 [🔄 重新提取与计算]")

    def _on_kalman_param_changed(self):
        if not self.analyzer.kalman_config:
            self.analyzer.kalman_config = copy.deepcopy(config.KALMAN_CONFIG)
        self.analyzer.kalman_config['max_coast_frames'] = self.sp_max_coast.value()
        self.analyzer.kalman_config['max_coast_time'] = self.sp_coast_time.value()
        self.analyzer.kalman_config['height_gate_limit'] = self.sp_gate_limit.value()
        self.analyzer.kalman_config['height_process_noise_q'] = self.sp_proc_noise.value()
        self.analyzer.kalman_config['height_measure_noise_r'] = self.sp_meas_noise.value()

        config.KALMAN_CONFIG['max_coast_frames'] = self.sp_max_coast.value()
        config.KALMAN_CONFIG['max_coast_time'] = self.sp_coast_time.value()
        config.KALMAN_CONFIG['height_gate_limit'] = self.sp_gate_limit.value()
        config.KALMAN_CONFIG['height_process_noise_q'] = self.sp_proc_noise.value()
        config.KALMAN_CONFIG['height_measure_noise_r'] = self.sp_meas_noise.value()

        self.invalidate_results()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121218;
                color: #e0e0e0;
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            }
            QPushButton {
                background-color: #262633;
                border: 1px solid #3d3d4f;
                border-radius: 4px;
                padding: 5px 12px;
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #38384a;
                border: 1px solid #66b2ff;
            }
            QProgressBar {
                background-color: #1e1e28;
                border: 1px solid #333344;
                border-radius: 3px;
                text-align: center;
                color: #ffffff;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #1e1e28;
                border: 1px solid #333344;
                border-radius: 3px;
                color: #ffffff;
                padding: 2px 5px;
            }
        """)

    def invalidate_results(self):
        self.points_data.clear()
        self.t_rel_list.clear()
        self.position_results.clear()
        for curve in self.position_curves + self.position_pred_curves:
            curve.clear()
        self.curve_low_meas.clear()
        self.curve_low_pred.clear()
        self.curve_high_meas.clear()
        self.curve_high_pred.clear()
        self.curve_diff.clear()
        self.curve_final_meas.clear()
        self.curve_final_pred.clear()
        self.p1.setTitle("小X端顶部 Z (待解算)", color='#66b2ff', size='11pt')
        self.p2.setTitle("大X端顶部 Z (待解算)", color='#f97316', size='11pt')
        self.lbl_status.setText("💡 参数已更新；点击 [🔄 重新提取与计算] 或正在自动更新时序曲线...")

    def start_computation(self, frames: List[TM16Frame], roi_detector: ShortEdgeDetector, cabin_detector: CabinDetector, height_config=None):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_recompute.setEnabled(False)
        self.lbl_status.setText("⏳ 正在统一计算通路执行逐帧扫描、圈车跟踪与时序滤波解算，请稍候...")

        self.worker = TimeSeriesComputeWorker(frames, roi_detector, cabin_detector, self.analyzer, height_config)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()

    @Slot(int, int)
    def _on_worker_progress(self, current: int, total: int):
        pct = int(current * 100.0 / total) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.lbl_progress.setText(f"{current} / {total} 帧 ({pct}%)")

    @Slot(dict)
    def _on_worker_finished(self, results: dict):
        self.progress_bar.setVisible(False)
        self.lbl_progress.setText("")
        self.btn_recompute.setEnabled(True)

        if not results:
            self.lbl_status.setText("⚠️ 计算完成，但未生成点集")
            return

        self.points_data = results.get('points', [])
        self.position_results = results.get('positions', [])
        self.t_rel_list = [p.t_rel for p in self.points_data]

        self._plot_results(results)
        self.results_ready.emit()

    @Slot(str)
    def _on_worker_failed(self, err_msg: str):
        self.progress_bar.setVisible(False)
        self.lbl_progress.setText("")
        self.btn_recompute.setEnabled(True)
        self.lbl_status.setText(f"❌ 计算失败: {err_msg}")
        QMessageBox.critical(self, "时序计算错误", f"后台扫描发生异常:\n{err_msg}")

    def _plot_results(self, res: dict):
        if not self.points_data:
            return

        t = np.array(self.t_rel_list)
        # 没有点云帧的真实时间空档不应被绘图连线掩盖。
        breaks = np.flatnonzero(np.diff(t) > self.analyzer.max_time_gap) + 1
        plot_t = np.insert(t, breaks, (t[breaks-1] + t[breaks]) * .5)

        def set_curve(curve, values):
            curve.setData(plot_t, np.insert(np.asarray(values, dtype=float), breaks, np.nan))
        zl_meas = np.array([p.z_low_filtered if p.point_type_low in ('reliable_measured', 'local_reacquired') else np.nan for p in self.points_data])
        zl_pred = np.array([p.z_low_filtered if p.point_type_low == 'predicted' else np.nan for p in self.points_data])

        zh_meas = np.array([p.z_high_filtered if p.point_type_high in ('reliable_measured', 'local_reacquired') else np.nan for p in self.points_data])
        zh_pred = np.array([p.z_high_filtered if p.point_type_high == 'predicted' else np.nan for p in self.points_data])

        diff = np.array([p.diff_filtered if p.diff_filtered is not None else np.nan for p in self.points_data])

        # 融合高度：区分两端全实测 vs 包含预测
        zf_meas = np.array([p.z_final if (p.point_type_low in ('reliable_measured', 'local_reacquired') and
                                           p.point_type_high in ('reliable_measured', 'local_reacquired'))
                            else np.nan for p in self.points_data])
        zf_pred = np.array([p.z_final if (p.point_type_low == 'predicted' or p.point_type_high == 'predicted')
                            else np.nan for p in self.points_data])

        set_curve(self.curve_low_meas, zl_meas)
        set_curve(self.curve_low_pred, zl_pred)
        set_curve(self.curve_high_meas, zh_meas)
        set_curve(self.curve_high_pred, zh_pred)
        set_curve(self.curve_diff, diff)
        set_curve(self.curve_final_meas, zf_meas)
        set_curve(self.curve_final_pred, zf_pred)
        for key, measured, predicted in zip(('x_high', 'x_low', 'cabin_length'),
                                            self.position_curves, self.position_pred_curves):
            set_curve(measured, [getattr(p, key) if p.position_type in ('reliable_measured', 'local_reacquired') else np.nan
                                for p in self.points_data])
            set_curve(predicted, [getattr(p, key) if p.position_type == 'predicted' else np.nan
                                for p in self.points_data])

        raw_l = res.get('raw_low_count', 0)
        raw_h = res.get('raw_high_count', 0)
        pass_l = res.get('pass_low_count', 0)
        pass_h = res.get('pass_high_count', 0)
        out_l = res.get('out_low_count', 0)
        out_h = res.get('out_high_count', 0)
        pred_l = res.get('predicted_low_count', 0)
        pred_h = res.get('predicted_high_count', 0)
        loc_l = res.get('local_reacquired_low_count', 0)
        loc_h = res.get('local_reacquired_high_count', 0)
        final_count = res.get('final_count', 0)

        mode_str = "卡尔曼滤波" if (self.analyzer.kalman_config and self.analyzer.kalman_config.get('enabled')) else "因果中值"
        self.p1.setTitle(f"小X端顶部 Z ({mode_str}) | 原始 {raw_l} 帧 → 质量通过 {pass_l} 帧 (局部恢复 {loc_l}) → 输出 {out_l} 帧 (预测 {pred_l})", color='#66b2ff', size='11pt')
        self.p2.setTitle(f"大X端顶部 Z ({mode_str}) | 原始 {raw_h} 帧 → 质量通过 {pass_h} 帧 (局部恢复 {loc_h}) → 输出 {out_h} 帧 (预测 {pred_h})", color='#f97316', size='11pt')

        real_points = [p for p in self.points_data if not p.is_gap_fill]
        real_final = sum(p.z_final is not None for p in real_points)
        cov = real_final * 100.0 / len(real_points) if real_points else 0.0
        self.lbl_status.setText(f"✅ 解算完成! 模式={mode_str}, 最终有效高度: {real_final} / {len(real_points)} 实际帧 "
                               f"(覆盖率 {cov:.1f}%), 时间空档预测点={len(self.points_data)-len(real_points)}, "
                               f"小X预测={pred_l}, 大X预测={pred_h}")

    def _on_plot_clicked(self, event):
        pos = event.scenePos()
        for p in self.plots:
            if p.sceneBoundingRect().contains(pos):
                mouse_point = p.vb.mapSceneToView(pos)
                clicked_t = mouse_point.x()
                self._scrub_to_time(clicked_t)
                break

    def _scrub_to_time(self, t: float):
        if not self.t_rel_list:
            return
        idx = int(np.argmin(np.abs(np.array(self.t_rel_list) - t)))
        self.set_current_point_idx(idx)

    def set_current_point_idx(self, pt_idx: int):
        if not self.points_data or pt_idx < 0 or pt_idx >= len(self.points_data):
            for line in self.cursor_lines:
                line.setVisible(False)
            return

        p = self.points_data[pt_idx]
        cur_t = p.t_rel
        for line in self.cursor_lines:
            line.setValue(cur_t)
            line.setVisible(True)

        zl_str = f"{p.z_low_filtered:.3f}m" if p.z_low_filtered is not None else "--"
        zh_str = f"{p.z_high_filtered:.3f}m" if p.z_high_filtered is not None else "--"
        diff_str = f"{p.diff_filtered:.3f}m" if p.diff_filtered is not None else "--"
        final_str = f"{p.z_final:.3f}m" if p.z_final is not None else "无效(缺测/超标)"

        position_str = (' | 前X={:.3f}m 后X={:.3f}m 长={:.3f}m [{}]'.format(
            p.x_high, p.x_low, p.cabin_length, p.position_type)
            if all(v is not None for v in (p.x_high, p.x_low, p.cabin_length)) else ' | 位置/车长无效')
        self.lbl_cursor_info.setText(
            f"帧 #{p.frame_idx+1} (t={cur_t:.2f}s) | 小X端={zl_str} [{p.point_type_low}:{p.low_reason}] | "
            f"大X端={zh_str} [{p.point_type_high}:{p.high_reason}] | 两端差={diff_str} | 最终={final_str}"
            + position_str
        )
        self.frame_selected.emit(p.frame_idx)

    def set_current_frame(self, frame_idx: int):
        """三维窗口选帧联动到最近的时序点。"""
        if not self.points_data:
            return
        for i, p in enumerate(self.points_data):
            if p.frame_idx == frame_idx and not p.is_gap_fill:
                self.set_current_point_idx(i)
                return

    def _export_image(self):
        if not self.points_data:
            QMessageBox.warning(self, "导出提示", "暂无曲线数据，请先提取计算。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存时序图", "TM16_Cabin_TimeSeries.png", "PNG Images (*.png);;All Files (*)")
        if path:
            exporter = pg.exporters.ImageExporter(self.tabs.currentWidget().scene())
            exporter.export(path)
            QMessageBox.information(self, "导出成功", f"时序图已保存至:\n{path}")
