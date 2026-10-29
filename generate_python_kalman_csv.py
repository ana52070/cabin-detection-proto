# -*- coding: utf-8 -*-
"""
生成标准 python_results_kalman.csv 脚本。
复用 cabin_timeseries.py 中的统一计算流水线 run_cabin_timeseries_pipeline，
确保 GUI、CSV 导出与验证脚本的一致性。
"""
import os
import csv
from pathlib import Path
import numpy as np

import config
from tm16_decoder import TM16Decoder
from short_edge_detector import ShortEdgeDetector
from cabin_detector import CabinDetector
from cabin_timeseries import run_cabin_timeseries_pipeline
from plc_adapter import compute_plc_measurements


def main():
    pcap_path = config.DEFAULT_TRUCK_PCAP
    print(f"Loading PCAP: {pcap_path}")
    frames = TM16Decoder().decode_pcap(pcap_path)
    print(f"Decoded {len(frames)} frames from PCAP.")

    roi_detector = ShortEdgeDetector()
    c_cfg = getattr(config, 'CABIN_DETECTOR_CONFIG', {})
    cabin_detector = CabinDetector(**c_cfg)

    print("Running unified pipeline...")
    res = run_cabin_timeseries_pipeline(
        frames=frames,
        roi_detector=roi_detector,
        cabin_detector=cabin_detector,
        tracking_config=config.CABIN_TRACKING_CONFIG,
        timeseries_config=config.HEIGHT_TIMESERIES_CONFIG,
        kalman_config=config.KALMAN_CONFIG
    )

    pts = res['points']
    csv_path = "python_results_kalman.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "t_rel", "x_low", "x_high", "cabin_length",
            "z_low_raw", "z_high_raw", "pass_low_qual", "pass_high_qual",
            "z_low_filtered", "z_high_filtered", "diff_filtered", "z_final",
            "point_type_low", "point_type_high",
            "time_since_reliable_low", "time_since_reliable_high",
            "low_reason", "high_reason", "is_gap_fill",
            "plc_head_position_mm", "plc_tail_position_mm",
            "plc_front_height_mm", "plc_rear_height_mm", "plc_valid_mask"
        ])

        for p in pts:
            plc = compute_plc_measurements(
                x_front=p.x_low, x_rear=p.x_high,
                z_front_raw=p.z_low_filtered, z_rear_raw=p.z_high_filtered,
                status="ok" if (p.z_low_filtered is not None and p.z_high_filtered is not None) else "invalid"
            )
            writer.writerow([
                p.frame_idx,
                round(p.t_rel, 6),
                round(p.x_low, 4) if p.x_low is not None else "",
                round(p.x_high, 4) if p.x_high is not None else "",
                round(p.cabin_length, 4) if p.cabin_length is not None else "",
                round(p.z_low_raw, 4) if p.z_low_raw is not None else "",
                round(p.z_high_raw, 4) if p.z_high_raw is not None else "",
                p.pass_low_qual,
                p.pass_high_qual,
                round(p.z_low_filtered, 4) if p.z_low_filtered is not None else "",
                round(p.z_high_filtered, 4) if p.z_high_filtered is not None else "",
                round(p.diff_filtered, 4) if p.diff_filtered is not None else "",
                round(p.z_final, 4) if p.z_final is not None else "",
                p.point_type_low,
                p.point_type_high,
                round(p.time_since_reliable_low, 4) if p.time_since_reliable_low is not None else "",
                round(p.time_since_reliable_high, 4) if p.time_since_reliable_high is not None else "",
                p.low_reason,
                p.high_reason,
                p.is_gap_fill,
                plc.head_position_mm if plc.head_position_mm is not None else "",
                plc.tail_position_mm if plc.tail_position_mm is not None else "",
                plc.front_height_mm if plc.front_height_mm is not None else "",
                plc.rear_height_mm if plc.rear_height_mm is not None else "",
                plc.valid_mask
            ])

    print(f"Exported {len(pts)} rows to {csv_path} with final_count={res['final_count']}")


if __name__ == "__main__":
    main()
