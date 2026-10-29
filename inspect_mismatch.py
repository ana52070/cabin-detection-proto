# -*- coding: utf-8 -*-
import pandas as pd

py = pd.read_csv('python_results_kalman.csv')
cpp = pd.read_csv('cpp_results_kalman.csv')
cpp['frame_matched'] = cpp['frame'] - 1
m = pd.merge(py, cpp, left_on='frame', right_on='frame_matched', suffixes=('_py', '_cpp'))
diff_x = (m['x_low_py'] - m['x_low_cpp']).abs()
bad = m[diff_x > 1.0]
print(f'Frames with x_low diff > 1.0m: {len(bad)}')
for _, r in bad.head(10).iterrows():
    f = r['frame_py']
    print(f"Frame {f}: py=({r['x_low_py']}, {r['x_high_py']}, len={r['cabin_length']}, st={r['status_py']}) vs cpp=({r['x_low_cpp']}, {r['x_high_cpp']}, len={r['cabin_length_m']}, st={r['status_cpp']})")
