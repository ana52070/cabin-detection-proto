# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

py = pd.read_csv('python_results.csv')
cpp = pd.read_csv('cpp_results.csv')

# 通过时间戳就近匹配 (最近时间差 < 0.05s)
py_times = py['timestamp'].values
cpp_times = cpp['time_ms'].values * 0.001 # 转换为秒，或者用相对时间对齐

# 采用 ROI 点数滑动窗口对齐
matched_pairs = []
py_idx = 0
cpp_idx = 0

while py_idx < len(py) and cpp_idx < len(cpp):
    p_pts = py.iloc[py_idx]['roi_points']
    c_pts = cpp.iloc[cpp_idx]['roi_points']
    
    if p_pts == c_pts:
        matched_pairs.append((py_idx, cpp_idx))
        py_idx += 1
        cpp_idx += 1
    else:
        # 寻找最近的匹配
        found = False
        for step in range(1, 5):
            if py_idx + step < len(py) and py.iloc[py_idx + step]['roi_points'] == c_pts:
                py_idx += step
                found = True
                break
            elif cpp_idx + step < len(cpp) and cpp.iloc[cpp_idx + step]['roi_points'] == p_pts:
                cpp_idx += step
                found = True
                break
        if not found:
            py_idx += 1
            cpp_idx += 1

print(f"Total points-matched frame pairs: {len(matched_pairs)} / {min(len(py), len(cpp))}")

diffs_len = []
diffs_xf = []
diffs_xr = []
status_agree = 0
both_ok_count = 0

for p_i, c_i in matched_pairs:
    pr = py.iloc[p_i]
    cr = cpp.iloc[c_i]
    if pr['status'] == cr['status']:
        status_agree += 1
    if pr['status'] == 'ok' and cr['status'] == 'ok':
        both_ok_count += 1
        diffs_len.append(abs(pr['cabin_length'] - cr['cabin_length_m']))
        diffs_xf.append(abs(pr['x_front'] - cr['x_low']))
        diffs_xr.append(abs(pr['x_rear'] - cr['x_high']))

diffs_len = np.array(diffs_len)
diffs_xf = np.array(diffs_xf)
diffs_xr = np.array(diffs_xr)

print(f"Status agreement rate: {status_agree / len(matched_pairs) * 100:.2f}% ({status_agree}/{len(matched_pairs)})")
print(f"Both OK frames: {both_ok_count}")
print(f"Cabin Length exact match (< 0.001m): {(diffs_len < 0.001).mean() * 100:.2f}%")
print(f"Cabin Length mean diff: {diffs_len.mean():.6f} m, max diff: {diffs_len.max():.6f} m")
print(f"X_front mean diff: {diffs_xf.mean():.6f} m, max diff: {diffs_xf.max():.6f} m")
print(f"X_rear  mean diff: {diffs_xr.mean():.6f} m, max diff: {diffs_xr.max():.6f} m")
