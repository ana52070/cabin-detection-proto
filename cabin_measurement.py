# -*- coding: utf-8 -*-
"""
车帮高度识别模块：
自底向上搜索垂直连续立面，遇到首个不连续垂直断档（ΔZ > max_z_gap）即停止，
取有效连续段顶部的 98% 分位数作为车帮高度。
支持不受 ROI Zmax 裁剪地使用完整点云高度。
"""
from collections import deque
from typing import Optional, Tuple, Dict, Any
import numpy as np
import config


def measure_continuous_board_height(
    points: np.ndarray,
    x_board: Optional[float],
    x_margin: float = 0.30,
    z_min: float = 0.50,
    z_max: float = 20.00,
    y_min: float = -3.00,
    y_max: float = 4.00,
    max_z_gap: float = 0.35,
    min_continuous_pts: int = 5,
    top_percentile: float = 98.0
) -> Tuple[Optional[float], int, str, np.ndarray]:
    """
    根据连续性自底向上搜索车帮高度：
    1. 在车帮位置 x_board 附近（±x_margin）提取点云；
    2. 从立面底部（z >= z_min）由低到高逐层搜索（按 Z 轴升序排列）；
       特别注意：不受 ROI 的 z_max 裁剪，z_max 默认开放至 20m；
    3. 检测相邻点间的垂直间隔 ΔZ，当 ΔZ > max_z_gap 时判定为垂直不连续断档；
    4. 分段找出第一个点数 >= min_continuous_pts 的有效连续立面段；
    5. 取该连续立面段顶部点的 top_percentile 分位数作为车帮高度；
    6. 返回 (height, count, reason, continuous_points)。
    """
    empty_pts = np.empty((0, 3), dtype=np.float32)
    if x_board is None or not np.isfinite(x_board) or len(points) == 0:
        return None, 0, 'invalid_input', empty_pts

    mask = (
        (np.abs(points[:, 0] - x_board) <= x_margin) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    wall_pts = points[mask]
    if len(wall_pts) < min_continuous_pts:
        return None, len(wall_pts), f'too_few_points ({len(wall_pts)} < {min_continuous_pts})', empty_pts

    sorted_idx = np.argsort(wall_pts[:, 2])
    z_sorted = wall_pts[sorted_idx, 2]

    diffs = np.diff(z_sorted)
    gap_breaks = np.where(diffs > max_z_gap)[0]

    # 切割为连续立面子段
    seg_starts = [0] + [b + 1 for b in gap_breaks]
    seg_ends = [b + 1 for b in gap_breaks] + [len(z_sorted)]

    chosen_seg_pts = None
    for s, e in zip(seg_starts, seg_ends):
        if (e - s) >= min_continuous_pts:
            chosen_seg_pts = wall_pts[sorted_idx[s:e]]
            break

    if chosen_seg_pts is None:
        return None, len(wall_pts), 'no_valid_continuous_segment', empty_pts

    h = float(np.percentile(chosen_seg_pts[:, 2], top_percentile))
    return h, len(chosen_seg_pts), 'ok', chosen_seg_pts


def board_heights(points, result, roi=None, height_config=None):
    """
    统一计算前后车帮高度（使用连续立面自底向上搜索算法）
    """
    if height_config is None:
        height_config = getattr(config, 'BOARD_HEIGHT_CONFIG', {})
    if roi is None:
        roi = getattr(config, 'ROI_CONFIG', {})

    x_margin = height_config.get('x_margin', 0.30)
    z_min = height_config.get('z_min', 0.50)
    # 车帮测高高度不应受到 ROI 的 z_max 截断，默认开放至 20m
    z_max = height_config.get('z_max', 20.00)
    y_min = roi.get('y_min', -3.00)
    y_max = roi.get('y_max', 4.00)
    max_z_gap = height_config.get('max_z_gap', 0.35)
    min_continuous_pts = height_config.get('min_continuous_pts', 5)
    top_percentile = height_config.get('top_percentile', 98.0)

    heights = {}
    for side, x in [('front', result.x_front), ('rear', result.x_rear)]:
        h, count, reason, chosen_pts = measure_continuous_board_height(
            points=points,
            x_board=x,
            x_margin=x_margin,
            z_min=z_min,
            z_max=z_max,
            y_min=y_min,
            y_max=y_max,
            max_z_gap=max_z_gap,
            min_continuous_pts=min_continuous_pts,
            top_percentile=top_percentile
        )
        heights[f'z_{side}_raw'] = h
        heights[f'z_{side}_pts'] = count
        heights[f'z_{side}_reason'] = reason
    return heights


class CabinMeasurementCollector:
    """工作线程和回放验证共用同一数据通路，预热样本不回写历史帧。"""

    def __init__(self, confirm_frames=3, roi=None, height_config=None):
        self.samples = deque(maxlen=confirm_frames)
        self.candidate_id = None
        self.roi = roi
        self.height_config = height_config

    def collect(self, frame_idx, t_rel, roi, result, tracker):
        row = dict(frame_idx=frame_idx, t_rel=t_rel, status=result.status,
                   x_front=result.x_front, x_rear=result.x_rear,
                   cabin_length=result.cabin_length, track_epoch=result.track_epoch,
                   track_event=result.track_event, candidate_id=tracker.candidate_id,
                   z_front_raw=None, z_rear_raw=None, warmup_samples=[])
        if result.status == 'ok':
            row.update(board_heights(roi, result, roi=self.roi, height_config=self.height_config))
        candidate = tracker.candidate
        if candidate is None:
            self.samples.clear()
            self.candidate_id = None
            return row
        if tracker.candidate_id != self.candidate_id:
            self.samples.clear()
            self.candidate_id = tracker.candidate_id
        sample = dict(frame_idx=frame_idx, t_rel=t_rel, **board_heights(roi, candidate, roi=self.roi, height_config=self.height_config))
        self.samples.append(sample)
        if tracker.confirmed_candidate_id == self.candidate_id:
            row['warmup_samples'] = [dict(s) for s in list(self.samples)[:-1]]
            self.samples.clear()
            self.candidate_id = None
        return row
