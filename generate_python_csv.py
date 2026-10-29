# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
from tm16_decoder import TM16Decoder, TM16Frame
from short_edge_detector import ShortEdgeDetector
from cabin_detector import CabinDetector, CabinPositionResult
from plc_adapter import compute_plc_measurements, PlcMeasurements
import config

def main():
    pcap_path = config.DEFAULT_TRUCK_PCAP
    print(f"Loading PCAP: {pcap_path}")
    if not os.path.isfile(pcap_path):
        print(f"Error: PCAP not found at {pcap_path}")
        return

    decoder = TM16Decoder()
    frames = decoder.decode_pcap(pcap_path)
    total_frames = len(frames)
    print(f"Decoded {total_frames} frames from PCAP.")

    roi_detector = ShortEdgeDetector()
    c_cfg = getattr(config, 'CABIN_DETECTOR_CONFIG', {})
    cabin_detector = CabinDetector(
        bin_size=c_cfg.get('bin_size', 0.17),
        min_peak_pts=c_cfg.get('min_peak_pts', 25),
        min_cabin_len=c_cfg.get('min_cabin_len', 6.00),
        max_cabin_len=c_cfg.get('max_cabin_len', 24.00),
        enable_z_uniformity=c_cfg.get('enable_z_uniformity', True),
        min_board_height=c_cfg.get('min_board_height', 1.80),
        max_empty_gap=c_cfg.get('max_empty_gap', 2.50),
        max_base_jump=c_cfg.get('max_base_jump', 1.00)
    )

    csv_path = "python_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "timestamp", "roi_points", "status",
            "x_front", "x_rear", "cabin_length",
            "pts_front", "pts_rear", "z_front_raw", "z_rear_raw",
            "z_check_status", "max_gap_found",
            "plc_head_position_mm", "plc_tail_position_mm",
            "plc_front_height_mm", "plc_rear_height_mm", "plc_valid_mask"
        ])

        for idx, frame in enumerate(frames):
            roi_pts = roi_detector.filter_roi(frame.points_site)
            res: CabinPositionResult = cabin_detector.detect(roi_pts)

            zf, zr = None, None
            if res.status == "ok":
                b1 = roi_pts[(roi_pts[:, 0] >= res.x_front - 0.2) & (roi_pts[:, 0] <= res.x_front + 0.2)]
                if len(b1) > 0:
                    zf = round(float(np.percentile(b1[:, 2], 98)), 4)
                b2 = roi_pts[(roi_pts[:, 0] >= res.x_rear - 0.2) & (roi_pts[:, 0] <= res.x_rear + 0.2)]
                if len(b2) > 0:
                    zr = round(float(np.percentile(b2[:, 2], 98)), 4)

            plc_meas = compute_plc_measurements(res.x_front, res.x_rear, zf, zr, res.status)

            writer.writerow([
                idx,
                round(frame.timestamp, 6),
                len(roi_pts),
                res.status,
                round(res.x_front, 4) if res.x_front is not None else "",
                round(res.x_rear, 4) if res.x_rear is not None else "",
                round(res.cabin_length, 4) if res.cabin_length else "",
                res.pts_front,
                res.pts_rear,
                zf if zf is not None else "",
                zr if zr is not None else "",
                res.z_check_status,
                round(res.max_gap_found, 4),
                plc_meas.head_position_mm if plc_meas.is_head_valid else "",
                plc_meas.tail_position_mm if plc_meas.is_tail_valid else "",
                plc_meas.front_height_mm if plc_meas.is_front_h_valid else "",
                plc_meas.rear_height_mm if plc_meas.is_rear_h_valid else "",
                plc_meas.valid_mask
            ])

            if (idx + 1) % 200 == 0 or idx == total_frames - 1:
                print(f"Processed frame {idx + 1}/{total_frames}...")

    print(f"Saved {total_frames} frames result to {csv_path}")

if __name__ == '__main__':
    main()
