# -*- coding: utf-8 -*-
import config
from tm16_decoder import TM16Decoder
from short_edge_detector import ShortEdgeDetector
from cabin_detector import CabinDetector
from cabin_tracking import CabinPositionTracker
from cabin_measurement import CabinMeasurementCollector
from cabin_kalman import KalmanConfig, PositionKalmanFilter, CabinHeightKalmanFilter
import numpy as np

frames = TM16Decoder().decode_pcap(config.DEFAULT_TRUCK_PCAP)
t0 = frames[0].timestamp
roi_filter = ShortEdgeDetector()

kf_cfg = KalmanConfig(enabled=True, max_coast_frames=5, max_coast_time=0.60)
pos_kf = PositionKalmanFilter(kf_cfg)
height_kf = CabinHeightKalmanFilter(kf_cfg, z_min=4.0, z_max=7.5, consistency_threshold=0.30)
tracker_kf = CabinPositionTracker(CabinDetector(**config.CABIN_DETECTOR_CONFIG), **config.CABIN_TRACKING_CONFIG)
collector_kf = CabinMeasurementCollector(tracker_kf.confirm_frames)

kalman_points = []
for i, f in enumerate(frames):
    t_rel = f.timestamp - t0
    roi = roi_filter.filter_roi(f.points_site)
    res = tracker_kf.update(roi, f.timestamp)
    if res.status == 'ok':
        pos_kf.update(res.x_front, res.x_rear, f.timestamp)
        row = collector_kf.collect(i, t_rel, roi, res, tracker_kf)
        z_front_raw = row.get('z_front_raw')
        z_rear_raw = row.get('z_rear_raw')
    else:
        coasted_pos = pos_kf.coast(f.timestamp)
        z_front_raw, z_rear_raw = None, None
        if coasted_pos is not None:
            xf_c, xr_c, _ = coasted_pos
            pf = roi[np.abs(roi[:, 0] - xf_c) <= 0.25]
            if len(pf) >= 5:
                top_f = float(np.percentile(pf[:, 2], 98))
                if 4.0 <= top_f <= 7.5: z_front_raw = top_f
            pr = roi[np.abs(roi[:, 0] - xr_c) <= 0.25]
            if len(pr) >= 5:
                top_r = float(np.percentile(pr[:, 2], 98))
                if 4.0 <= top_r <= 7.5: z_rear_raw = top_r

    zf, zr, diff, z_final = height_kf.process(z_front_raw, z_rear_raw, f.timestamp)
    kalman_points.append(dict(frame_idx=i, t_rel=t_rel, zf=zf, zr=zr, diff=diff, z_final=z_final,
                              z_front_raw=z_front_raw, z_rear_raw=z_rear_raw))

valid_kf = [p for p in kalman_points if p['z_final'] is not None]
for a, b in zip(valid_kf, valid_kf[1:]):
    if b['frame_idx'] == a['frame_idx'] + 1 and 0 < b['t_rel'] - a['t_rel'] <= 0.3:
        step = abs(b['z_final'] - a['z_final'])
        if step > 0.15:
            fa, fb = a['frame_idx'], b['frame_idx']
            print(f"Jump {step*100:.2f}cm at frames {fa}->{fb}, t={b['t_rel']:.2f}s, prev={a['z_final']:.3f}, curr={b['z_final']:.3f}")
            print(f"  Frame {fa}: zf={a['zf']}, zr={a['zr']}, raw=({a['z_front_raw']}, {a['z_rear_raw']})")
            print(f"  Frame {fb}: zf={b['zf']}, zr={b['zr']}, raw=({b['z_front_raw']}, {b['z_rear_raw']})")
