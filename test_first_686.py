# -*- coding: utf-8 -*-
import pandas as pd

py = pd.read_csv('python_results.csv')
cpp = pd.read_csv('cpp_results.csv')

N = 686
py_sub = py.iloc[:N]
cpp_sub = cpp.iloc[:N]

print(f"Testing first {N} frames (0 to {N-1}):")
exact_pts = (py_sub['roi_points'].values == cpp_sub['roi_points'].values).mean() * 100
print(f"ROI points 100% exact match: {exact_pts:.2f}%")

# 对比车厢长度和位置
py_ok = (py_sub['status'] == 'ok')
cpp_ok = (cpp_sub['status'] == 'ok')

print(f"Python ok frames in first {N}: {py_ok.sum()}")
print(f"C++ status counts in first {N}:")
print(cpp_sub['status'].value_counts())

# C++ 中有 x_low 值的帧
cpp_has_pos = cpp_sub['x_low'].notna()
both_pos = py_ok & cpp_has_pos
print(f"Both detected cabin position: {both_pos.sum()} / {py_ok.sum()}")

diff_x_front = (py_sub.loc[both_pos, 'x_front'] - cpp_sub.loc[both_pos, 'x_low']).abs()
diff_x_rear = (py_sub.loc[both_pos, 'x_rear'] - cpp_sub.loc[both_pos, 'x_high']).abs()
diff_len = (py_sub.loc[both_pos, 'cabin_length'] - cpp_sub.loc[both_pos, 'cabin_length_m']).abs()

print(f"x_front max diff: {diff_x_front.max():.6f} m, mean diff: {diff_x_front.mean():.6f} m")
print(f"x_rear  max diff: {diff_x_rear.max():.6f} m, mean diff: {diff_x_rear.mean():.6f} m")
print(f"length  max diff: {diff_len.max():.6f} m, mean diff: {diff_len.mean():.6f} m")
print(f"Length diff == 0.0000m (exact match): {(diff_len < 1e-4).mean() * 100:.2f}%")
