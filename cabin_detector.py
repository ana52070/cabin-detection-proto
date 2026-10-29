# -*- coding: utf-8 -*-
"""
Cabin Position Detector:
使用 X 轴粗聚类（圈车）的结果作为前后车帮位置：
- 大X端为前车帮 / 车头 (x_front)
- 小X端为后车帮 / 车尾 (x_rear)
- 车厢长度为粗聚类跨度 (cabin_length = cluster_span)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import config

@dataclass
class CabinPositionResult:
    status: str                         # "ok", "no_points", "insufficient_peaks", "no_valid_pair", "no_clusters"
    x_front: Optional[float] = None     # Front board X (meters, 粗聚类大X端 / 车头)
    x_rear: Optional[float] = None      # Rear board X (meters, 粗聚类小X端 / 车尾)
    cabin_length: float = 0.0           # Length in meters (|x_rear - x_front|)
    pts_front: int = 0                  # Point count at front board
    pts_rear: int = 0                   # Point count at rear board
    cabin_box: Optional[Tuple[float, float, float, float, float, float]] = None # (x_min, x_max, y_min, y_max, z_min, z_max)
    bin_centers: Optional[np.ndarray] = None
    hist_counts: Optional[np.ndarray] = None
    z_check_status: str = "disabled"    # "pass", "not_vertical_board", "gap_too_large", "base_jump_too_large", "disabled"
    max_gap_found: float = 0.0          # Max empty gap found between boards (meters)
    track_epoch: int = 0               # 跟踪目标代次；单帧检测为0
    track_event: str = 'single_frame'   # 正常跟踪、同目标恢复或新目标捕获


class CabinDetector:
    """
    Cabin Detector using X-axis coarse clustering results as front/rear cabin boards.
    """
    def __init__(self,
                 roi: Optional[dict] = None,
                 roi_config: Optional[dict] = None,
                 bin_size: float = 0.17,
                 min_peak_pts: int = 25,
                 min_cabin_len: float = 6.00,
                 max_cabin_len: float = 24.00,
                 enable_z_uniformity: bool = True,
                 min_board_height: float = 1.80,
                 max_empty_gap: float = 2.50,
                 max_base_jump: float = 1.00):
        eff = roi if roi is not None else roi_config
        self.roi = dict(eff if eff is not None else config.ROI_CONFIG)
        self.bin_size = bin_size
        self.min_peak_pts = min_peak_pts
        self.min_cabin_len = min_cabin_len
        self.max_cabin_len = max_cabin_len
        self.enable_z_uniformity = enable_z_uniformity
        self.min_board_height = min_board_height
        self.max_empty_gap = max_empty_gap
        self.max_base_jump = max_base_jump

    def detect(self,
               roi_points: np.ndarray,
               reference: Optional[CabinPositionResult] = None,
               max_position_step: float = 0.8,
               length_tolerance: float = 0.6,
               roi: Optional[dict] = None,
               coarse_res: Optional[Any] = None) -> CabinPositionResult:
        """
        前后车帮位置由【X轴网格粗聚类（圈车）】结果决定：
        - 粗聚类大X端为前车帮/车头 (x_front)
        - 粗聚类小X端为后车帮/车尾 (x_rear)
        - 车厢净长为粗聚类跨度 (cabin_length = cluster_span)
        - 车厢三维包围框对齐粗聚类 X 范围与 ROI Y/Z 范围
        """
        r = roi if roi is not None else self.roi
        y_min = float(r['y_min'])
        y_max = float(r['y_max'])
        z_min = float(r['z_min'])
        z_max = float(r['z_max'])

        if len(roi_points) == 0:
            return CabinPositionResult(
                status="no_points",
                x_front=None,
                x_rear=None,
                cabin_length=0.0,
                pts_front=0,
                pts_rear=0,
                cabin_box=None,
                z_check_status="无ROI点云",
                max_gap_found=0.0
            )

        if coarse_res is None:
            from short_edge_detector import ShortEdgeDetector
            sed = getattr(self, 'coarse_detector', None) or ShortEdgeDetector(roi_config=r)
            coarse_res = sed.detect_coarse(roi_points)

        if coarse_res.status != "ok" or coarse_res.coarse_x is None:
            return CabinPositionResult(
                status=coarse_res.status if coarse_res.status != "ok" else "no_cluster",
                x_front=None,
                x_rear=None,
                cabin_length=0.0,
                pts_front=0,
                pts_rear=0,
                cabin_box=None,
                z_check_status=f"粗聚类未圈出车辆 ({coarse_res.status})",
                max_gap_found=0.0
            )

        c_min, c_max = coarse_res.coarse_x
        x_rear = float(min(c_min, c_max))   # 小X端为后车帮 (车尾)
        x_front = float(max(c_min, c_max))  # 大X端为前车帮 (车头)
        cabin_len = float(coarse_res.cluster_span)
        box = (x_rear, x_front, y_min, y_max, z_min, z_max)

        # 可见跨度缩短并不表示真实车长改变；触及ROI边界时完整长度未知。
        if x_rear <= r['x_min'] + 1e-6 or x_front >= r['x_max'] - 1e-6:
            return CabinPositionResult('partial_view', x_front=x_front, x_rear=x_rear,
                                       cabin_length=cabin_len, cabin_box=box,
                                       z_check_status='车体被ROI边界截断')
        if reference is not None:
            if (abs(x_front-reference.x_front) > max_position_step
                    or abs(x_rear-reference.x_rear) > max_position_step
                    or abs(cabin_len-reference.cabin_length) > length_tolerance):
                return CabinPositionResult('position_rejected', z_check_status='端点位移或长度变化超限')

        margin = 0.35
        pts_front = int(np.sum(np.abs(roi_points[:, 0] - x_front) <= margin))
        pts_rear = int(np.sum(np.abs(roi_points[:, 0] - x_rear) <= margin))

        return CabinPositionResult(
            status="ok",
            x_front=x_front,
            x_rear=x_rear,
            cabin_length=cabin_len,
            pts_front=pts_front,
            pts_rear=pts_rear,
            cabin_box=box,
            z_check_status="pass (X轴粗聚类圈车定位)",
            max_gap_found=0.0
        )

    def recover_local(self, points, reference, search_radius=.65, length_tolerance=.6):
        """在参考端点附近分别找本帧立面；不要求中间点云连通，不补造缺失端点。"""
        from short_edge_detector import ShortEdgeDetector
        sed = getattr(self, 'coarse_detector', None) or ShortEdgeDetector(roi_config=self.roi)
        coarse = sed.detect_coarse(points)
        step = (sed.roi['x_max']-sed.roi['x_min']) / len(coarse.counts)
        recovered = []
        for x, low in ((reference.x_rear, True), (reference.x_front, False)):
            centers = sed.roi['x_min'] + (np.arange(len(coarse.counts))+.5)*step
            candidates = np.flatnonzero((np.abs(centers-x) <= search_radius) &
                                        (coarse.counts >= sed.cluster_min_points))
            if not len(candidates):
                return CabinPositionResult('local_missing')
            # 先找靠近预测的连续占用区，避免把搜索窗口边缘的车体内部点当端点。
            runs = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1)+1)
            run = min(runs, key=lambda v: np.min(abs(centers[v]-x)))
            idx = run[0] if low else run[-1]
            endpoint = sed.roi['x_min'] + (idx if low else idx+1)*step
            wall = coarse.roi_points[np.abs(coarse.roi_points[:, 0]-endpoint) <= .65]
            if len(wall) < 8:
                return CabinPositionResult('local_missing')
            p5, p98 = np.percentile(wall[:, 2], [5, 98])
            if p98-p5 < .9:
                return CabinPositionResult('local_missing')
            recovered.append(float(endpoint))
        low, high = recovered
        length = high-low
        if (low <= self.roi['x_min']+1e-6 or high >= self.roi['x_max']-1e-6
                or length <= 0 or abs(length-reference.cabin_length) > length_tolerance):
            return CabinPositionResult('local_rejected')
        return CabinPositionResult('ok', x_front=high, x_rear=low, cabin_length=length,
                                   cabin_box=(low, high, self.roi['y_min'], self.roi['y_max'],
                                              self.roi['z_min'], self.roi['z_max']),
                                   z_check_status='两端本帧立面局部恢复')

    def check_z_uniformity_and_continuity(self,
                                          roi_points: np.ndarray,
                                          x1: float,
                                          x2: float,
                                          bin_edges_1: Tuple[float, float],
                                          bin_edges_2: Tuple[float, float]) -> Tuple[bool, str, float]:
        """
        师傅限定值检验：
        1. 检查两候选峰本身是否为高耸垂直立面（车帮结构），排除平顶车头盖板；
        2. 检查两峰之间底板是否连续，有无突然大断档。
        """
        pts_b1 = roi_points[(roi_points[:, 0] >= bin_edges_1[0] - 0.1) & (roi_points[:, 0] <= bin_edges_1[1] + 0.1)]
        pts_b2 = roi_points[(roi_points[:, 0] >= bin_edges_2[0] - 0.1) & (roi_points[:, 0] <= bin_edges_2[1] + 0.1)]

        span_1 = float(pts_b1[:, 2].max() - pts_b1[:, 2].min()) if len(pts_b1) > 0 else 0.0
        span_2 = float(pts_b2[:, 2].max() - pts_b2[:, 2].min()) if len(pts_b2) > 0 else 0.0

        if span_1 < self.min_board_height or span_2 < self.min_board_height:
            return False, f"非垂直车帮立面 (Z跨度 {min(span_1, span_2):.2f}m < {self.min_board_height}m)", 0.0

        x_min, x_max = min(x1, x2), max(x1, x2)
        step = 0.40 # 40cm sampling slice
        sample_xs = np.arange(x_min + 0.35, x_max - 0.35, step)
        if len(sample_xs) < 2:
            return True, "pass (span too short)", 0.0

        consecutive_gap = 0.0
        max_gap = 0.0
        z_bases = []

        for sx in sample_xs:
            in_slice = (roi_points[:, 0] >= sx) & (roi_points[:, 0] < sx + step)
            pts_s = roi_points[in_slice]
            if len(pts_s) == 0:
                consecutive_gap += step
                max_gap = max(max_gap, consecutive_gap)
            else:
                consecutive_gap = 0.0
                z_base = float(np.percentile(pts_s[:, 2], 10))
                z_bases.append((sx, z_base))

        if max_gap > self.max_empty_gap:
            return False, f"空隙断档过大 ({max_gap:.2f}m > {self.max_empty_gap}m)", max_gap

        if len(z_bases) >= 2:
            base_diffs = np.abs(np.diff([b[1] for b in z_bases]))
            if len(base_diffs) > 0 and np.max(base_diffs) > self.max_base_jump:
                return False, f"底板Z轴突变断档 ({np.max(base_diffs):.2f}m > {self.max_base_jump}m)", max_gap

        return True, "pass", max_gap

    def detect_histogram(self, roi_points: np.ndarray,
                         reference: Optional[CabinPositionResult] = None,
                         max_position_step: float = 0.8,
                         length_tolerance: float = 0.6) -> CabinPositionResult:
        """
        Runs X-histogram voting with mentor's vertical board and Z-uniformity constraint.
        """
        if len(roi_points) == 0:
            return CabinPositionResult(status="no_points")

        x_coords = roi_points[:, 0]
        x_min, x_max = float(x_coords.min()), float(x_coords.max())

        span = x_max - x_min
        if span < self.min_cabin_len * 0.5:
            return CabinPositionResult(status="span_too_short")

        bin_step = max(0.02, float(self.bin_size))
        bins = np.arange(x_min, x_max + bin_step, bin_step)
        if len(bins) < 2:
            return CabinPositionResult(status="insufficient_bins")

        hist, bin_edges = np.histogram(x_coords, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

        peaks: List[int] = []
        for i in range(len(hist)):
            cnt = hist[i]
            if cnt >= self.min_peak_pts:
                left = (i == 0 or cnt >= hist[i - 1])
                right = (i == len(hist) - 1 or cnt >= hist[i + 1])
                if left and right:
                    peaks.append(i)

        merged_peaks = []
        if peaks:
            group = [peaks[0]]
            for idx in peaks[1:]:
                if idx - group[-1] <= 2:
                    group.append(idx)
                else:
                    merged_peaks.append(max(group, key=lambda k: hist[k]))
                    group = [idx]
            if group:
                merged_peaks.append(max(group, key=lambda k: hist[k]))

        if len(merged_peaks) < 2:
            top_sorted = np.argsort(hist)[::-1]
            candidates = [int(k) for k in top_sorted if hist[k] >= max(5, self.min_peak_pts // 2)][:15]
        else:
            candidates = merged_peaks

        if len(candidates) < 2:
            return CabinPositionResult(
                status="insufficient_peaks",
                bin_centers=bin_centers,
                hist_counts=hist
            )

        pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                idx_a, idx_b = candidates[i], candidates[j]
                xa, xb = float(bin_centers[idx_a]), float(bin_centers[idx_b])
                dist = abs(xa - xb)
                if self.min_cabin_len <= dist <= self.max_cabin_len:
                    score = int(hist[idx_a] + hist[idx_b])
                    pairs.append((idx_a, idx_b, min(xa, xb), max(xa, xb), dist, score))

        if not pairs:
            return CabinPositionResult(
                status="no_valid_pair",
                bin_centers=bin_centers,
                hist_counts=hist
            )

        if reference is not None:
            pairs = [p for p in pairs
                     if abs(p[2] - reference.x_rear) <= max_position_step
                     and abs(p[3] - reference.x_front) <= max_position_step
                     and abs(p[4] - reference.cabin_length) <= length_tolerance]
            if not pairs:
                return CabinPositionResult(
                    status="tracking_lost",
                    bin_centers=bin_centers,
                    hist_counts=hist
                )

        selected_pair = None
        max_gap_val = 0.0
        z_status = "disabled"

        if self.enable_z_uniformity:
            valid_pairs = []
            for p in pairs:
                idx_a, idx_b, xa, xb, dist, score = p
                edge_a = (float(bin_edges[idx_a]), float(bin_edges[idx_a + 1]))
                edge_b = (float(bin_edges[idx_b]), float(bin_edges[idx_b + 1]))

                passed, reason, max_g = self.check_z_uniformity_and_continuity(roi_points, xa, xb, edge_a, edge_b)
                if passed:
                    valid_pairs.append((p, max_g))

            if not valid_pairs:
                return CabinPositionResult(
                    status="z_breakage_rejected",
                    bin_centers=bin_centers,
                    hist_counts=hist,
                    z_check_status="未通过立面高度或底板均匀性检验"
                )

            valid_pairs.sort(key=lambda item: item[0][5], reverse=True)
            selected_pair = valid_pairs[0][0]
            max_gap_val = valid_pairs[0][1]
            z_status = "pass"
        else:
            pairs.sort(key=lambda p: p[5], reverse=True)
            selected_pair = pairs[0]
            z_status = "disabled"

        idx_low = selected_pair[0] if bin_centers[selected_pair[0]] < bin_centers[selected_pair[1]] else selected_pair[1]
        idx_high = selected_pair[1] if bin_centers[selected_pair[0]] < bin_centers[selected_pair[1]] else selected_pair[0]

        x_rear = float(bin_centers[idx_low])   # 小X端 (后车帮)
        x_front = float(bin_centers[idx_high]) # 大X端 (前车帮)
        cabin_len = float(abs(x_front - x_rear))
        pts_rear = int(hist[idx_low])
        pts_front = int(hist[idx_high])

        margin = bin_step * 1.5
        cabin_pts_mask = (roi_points[:, 0] >= x_rear - margin) & (roi_points[:, 0] <= x_front + margin)
        cabin_pts = roi_points[cabin_pts_mask]

        if len(cabin_pts) >= 10:
            y_coords = cabin_pts[:, 1]
            z_coords = cabin_pts[:, 2]
            y_min = float(np.percentile(y_coords, 1))
            y_max = float(np.percentile(y_coords, 99))
            z_min = float(np.percentile(z_coords, 0.5))
            z_max = float(np.percentile(z_coords, 99.5))
        else:
            y_min = float(config.ROI_CONFIG['y_min'])
            y_max = float(config.ROI_CONFIG['y_max'])
            z_min = float(config.ROI_CONFIG['z_min'])
            z_max = float(config.ROI_CONFIG['z_max'])

        box = (x_rear, x_front, y_min, y_max, z_min, z_max)

        return CabinPositionResult(
            status="ok",
            x_front=x_front,
            x_rear=x_rear,
            cabin_length=cabin_len,
            pts_front=pts_front,
            pts_rear=pts_rear,
            cabin_box=box,
            bin_centers=bin_centers,
            hist_counts=hist,
            z_check_status=z_status,
            max_gap_found=max_gap_val
        )
