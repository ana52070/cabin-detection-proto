# -*- coding: utf-8 -*-
import pandas as pd

py = pd.read_csv('python_results.csv')
cpp = pd.read_csv('cpp_results.csv')
cpp['frame_matched'] = cpp['frame'] - 1
merged = pd.merge(py, cpp, left_on='frame', right_on='frame_matched', suffixes=('_py', '_cpp'))

diff_frames = merged[(merged['status_py'] == 'ok') & (merged['x_low'].notna()) & ((merged['cabin_length'] - merged['cabin_length_m']).abs() > 0.05)]
print('Number of diff frames:', len(diff_frames))
for _, r in diff_frames.head(10).iterrows():
    f = r['frame_py']
    pl = r['cabin_length']
    cl = r['cabin_length_m']
    pxf = r['x_front']
    pxr = r['x_rear']
    cxf = r['x_low']
    cxr = r['x_high']
    print(f"Frame {f:4d}: Py len={pl:6.2f} [{pxf:6.2f}, {pxr:6.2f}] | Cpp len={cl:6.2f} [{cxf:6.2f}, {cxr:6.2f}]")
