# -*- coding: utf-8 -*-
from tm16_decoder import TM16Decoder
from short_edge_detector import ShortEdgeDetector
from cabin_detector import CabinDetector
import config
import numpy as np

dec = TM16Decoder()
frames = dec.decode_pcap(config.DEFAULT_TRUCK_PCAP)

roi_detector = ShortEdgeDetector()
cabin_detector = CabinDetector(**config.CABIN_DETECTOR_CONFIG)

for f_idx in [715, 716, 717, 718, 719, 720]:
    frame = frames[f_idx]
    roi_pts = roi_detector.filter_roi(frame.points_site)
    res = cabin_detector.detect(roi_pts)
    print(f"\nFrame {f_idx}: status={res.status}, x_front={res.x_front:.2f}, x_rear={res.x_rear:.2f}, len={res.cabin_length:.2f}")
    if res.status == 'ok':
        b1 = roi_pts[(roi_pts[:, 0] >= res.x_front - 0.2) & (roi_pts[:, 0] <= res.x_front + 0.2)]
        b2 = roi_pts[(roi_pts[:, 0] >= res.x_rear - 0.2) & (roi_pts[:, 0] <= res.x_rear + 0.2)]
        z1_98 = np.percentile(b1[:, 2], 98) if len(b1) > 0 else 0
        z2_98 = np.percentile(b2[:, 2], 98) if len(b2) > 0 else 0
        print(f"  Front (low_x={res.x_front:.2f}): pts={len(b1)}, z_min={b1[:,2].min():.2f}, z_max={b1[:,2].max():.2f}, z_98={z1_98:.2f}")
        print(f"  Rear  (high_x={res.x_rear:.2f}): pts={len(b2)}, z_min={b2[:,2].min():.2f}, z_max={b2[:,2].max():.2f}, z_98={z2_98:.2f}")
