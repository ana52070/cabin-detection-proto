# -*- coding: utf-8 -*-
from tm16_decoder import TM16Decoder
from short_edge_detector import ShortEdgeDetector
from cabin_detector import CabinDetector
import config
import numpy as np

decoder = TM16Decoder()
frames = decoder.decode_pcap(config.DEFAULT_TRUCK_PCAP)

f_idx = 686
frame = frames[f_idx]
roi_detector = ShortEdgeDetector()
roi_pts = roi_detector.filter_roi(frame.points_site)

c_cfg = config.CABIN_DETECTOR_CONFIG
detector = CabinDetector(**c_cfg)

# 我们把 detector.detect 的中间过程完整打印出来
x_coords = roi_pts[:, 0]
x_min, x_max = float(x_coords.min()), float(x_coords.max())
bin_step = max(0.02, float(detector.bin_size))
bins = np.arange(x_min, x_max + bin_step, bin_step)
hist, bin_edges = np.histogram(x_coords, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

print(f"Frame {f_idx}: x_min={x_min:.4f}, x_max={x_max:.4f}, num_bins={len(hist)}")

# 峰值
peaks = []
for i in range(len(hist)):
    cnt = hist[i]
    if cnt >= detector.min_peak_pts:
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

print(f"Merged peaks count: {len(merged_peaks)}")
for p in merged_peaks:
    print(f"  Peak bin {p:3d}: x={bin_centers[p]:6.2f}, pts={hist[p]}")

pairs = []
for i in range(len(merged_peaks)):
    for j in range(i + 1, len(merged_peaks)):
        ia, ib = merged_peaks[i], merged_peaks[j]
        xa, xb = float(bin_centers[ia]), float(bin_centers[ib])
        dist = abs(xa - xb)
        if detector.min_cabin_len <= dist <= detector.max_cabin_len:
            score = int(hist[ia] + hist[ib])
            pairs.append((ia, ib, min(xa, xb), max(xa, xb), dist, score))

pairs.sort(key=lambda p: p[5], reverse=True)
print("\nCandidate pairs (sorted by score):")
for p in pairs[:10]:
    ia, ib, xa, xb, dist, score = p
    e1 = (bin_edges[ia], bin_edges[ia+1])
    e2 = (bin_edges[ib], bin_edges[ib+1])
    passed, reason, cur_gap = detector.check_z_uniformity_and_continuity(roi_pts, xa, xb, e1, e2)
    print(f"  [{xa:6.2f}, {xb:6.2f}] dist={dist:6.2f}m, score={score:3d} (pts={hist[ia]}+{hist[ib]}), check={passed} ({reason})")
