# -*- coding: utf-8 -*-
import sys
import copy
import numpy as np

import config
from tm16_decoder import TM16Decoder
from short_edge_detector import ShortEdgeDetector
from cabin_detector import CabinDetector
from cabin_tracking import CabinPositionTracker
from cabin_measurement import CabinMeasurementCollector
from cabin_timeseries import CabinTimeSeriesAnalyzer
from cabin_kalman import KalmanConfig, PositionKalmanFilter, CabinHeightKalmanFilter

def evaluate():
    print("Loading PCAP...")
    frames = TM16Decoder().decode_pcap(config.DEFAULT_TRUCK_PCAP)
    print(f"Decoded {len(frames)} frames.")
    t0 = frames[0].timestamp

    roi_filter = ShortEdgeDetector()

    # 1. Baseline Run (Without Kalman)
    print("\n--- Running Baseline (No Kalman) ---")
    tracker_base = CabinPositionTracker(CabinDetector(**config.CABIN_DETECTOR_CONFIG), **config.CABIN_TRACKING_CONFIG)
    collector_base = CabinMeasurementCollector(tracker_base.confirm_frames)
    after_base = []
    for i, f in enumerate(frames):
        roi = roi_filter.filter_roi(f.points_site)
        res = tracker_base.update(roi, f.timestamp)
        after_base.append(collector_base.collect(i, f.timestamp - t0, roi, res, tracker_base))

    analyzer_base = CabinTimeSeriesAnalyzer(**config.HEIGHT_TIMESERIES_CONFIG)
    res_base = analyzer_base.analyze_frames(after_base)
    pts_base = res_base['points']
    valid_base = [p for p in pts_base if p.z_final is not None]
    print(f"Baseline valid output frames: {len(valid_base)} / {len(pts_base)}")

    # 2. Kalman-Enhanced Run
    print("\n--- Running With Kalman Filter ---")
    kf_cfg = KalmanConfig(
        enabled=True,
        max_coast_frames=5,
        max_coast_time=0.60,
        pos_process_noise_q=0.05,
        pos_measure_noise_r=0.15,
        height_process_noise_q=0.001,
        height_measure_noise_r=0.03,
        height_init_cov=0.10
    )

    pos_kf = PositionKalmanFilter(kf_cfg)
    height_kf = CabinHeightKalmanFilter(kf_cfg, z_min=4.0, z_max=7.5, consistency_threshold=0.30)

    tracker_kf = CabinPositionTracker(CabinDetector(**config.CABIN_DETECTOR_CONFIG), **config.CABIN_TRACKING_CONFIG)
    collector_kf = CabinMeasurementCollector(tracker_kf.confirm_frames)

    kalman_points = []
    for i, f in enumerate(frames):
        t_rel = f.timestamp - t0
        roi = roi_filter.filter_roi(f.points_site)
        res = tracker_kf.update(roi, f.timestamp)

        # 如果单帧检测成功且处于跟踪状态
        if res.status == 'ok':
            pos_kf.update(res.x_front, res.x_rear, f.timestamp)
            row = collector_kf.collect(i, t_rel, roi, res, tracker_kf)
            z_front_raw = row.get('z_front_raw')
            z_rear_raw = row.get('z_rear_raw')
        else:
            # 缺失帧或漏检帧：尝试利用卡尔曼外推
            coasted_pos = pos_kf.coast(f.timestamp)
            z_front_raw = None
            z_rear_raw = None
            if coasted_pos is not None:
                xf_coasted, xr_coasted, len_coasted = coasted_pos
                # 检查点云中是否有局部的车帮点（小漏检恢复）
                # 大X端(前)
                pts_front = roi[np.abs(roi[:, 0] - xf_coasted) <= 0.25]
                if len(pts_front) >= 5:
                    top_f = float(np.percentile(pts_front[:, 2], 98))
                    if 4.0 <= top_f <= 7.5:
                        z_front_raw = top_f
                # 小X端(后)
                pts_rear = roi[np.abs(roi[:, 0] - xr_coasted) <= 0.25]
                if len(pts_rear) >= 5:
                    top_r = float(np.percentile(pts_rear[:, 2], 98))
                    if 4.0 <= top_r <= 7.5:
                        z_rear_raw = top_r

        zf, zr, diff, z_final = height_kf.process(z_front_raw, z_rear_raw, f.timestamp)
        kalman_points.append({
            'frame_idx': i,
            't_rel': t_rel,
            'z_front_raw': z_front_raw,
            'z_rear_raw': z_rear_raw,
            'z_front_kf': zf,
            'z_rear_kf': zr,
            'diff_kf': diff,
            'z_final': z_final
        })

    valid_kf = [p for p in kalman_points if p['z_final'] is not None]
    print(f"Kalman valid output frames: {len(valid_kf)} / {len(kalman_points)}")
    print(f"Gain: +{len(valid_kf) - len(valid_base)} valid frames (+{(len(valid_kf) - len(valid_base))/len(valid_base)*100:.1f}%)")

    # 统计连续跳变 (jumps > 0.15m)
    steps = [abs(b['z_final'] - a['z_final'])
             for a, b in zip(valid_kf, valid_kf[1:])
             if b['frame_idx'] == a['frame_idx'] + 1 and 0 < b['t_rel'] - a['t_rel'] <= 0.3]
    jumps_15 = sum(s > 0.15 for s in steps)
    max_step = max(steps) if steps else 0.0
    print(f"Jumps > 15cm: {jumps_15}, Max consecutive step: {max_step*100:.2f} cm")

    # 统计内部 gaps
    gaps = []
    in_gap = False
    gap_start = 0
    for i, p in enumerate(kalman_points):
        if p['z_final'] is None:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                in_gap = False
                gaps.append((gap_start, i - 1, i - gap_start, kalman_points[gap_start]['t_rel'], kalman_points[i-1]['t_rel']))

    first_i = next(i for i, p in enumerate(kalman_points) if p['z_final'] is not None)
    last_i = [i for i, p in enumerate(kalman_points) if p['z_final'] is not None][-1]
    internal_gaps = [g for g in gaps if g[0] > first_i and g[1] < last_i]
    print(f"Internal gaps count: {len(internal_gaps)} (Baseline was > 50 gaps)")
    for g in internal_gaps:
        print(f"  Remaining Gap: frames {g[0]}-{g[1]} ({g[2]} frames, dt={g[4]-g[3]:.3f}s, t={g[3]:.2f}s..{g[4]:.2f}s)")

if __name__ == '__main__':
    evaluate()
