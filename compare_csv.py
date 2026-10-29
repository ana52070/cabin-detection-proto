# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def compare(py_csv_path, cpp_csv_path):
    print(f"Loading Python CSV: {py_csv_path}")
    py_df = pd.read_csv(py_csv_path)

    print(f"Loading C++ CSV: {cpp_csv_path}")
    cpp_df = pd.read_csv(cpp_csv_path)

    print(f"Python total rows: {len(py_df)}, C++ total rows: {len(cpp_df)}")

    py_first_frame = py_df.iloc[0]['frame']
    cpp_first_frame = cpp_df.iloc[0]['frame']
    print(f"Python first frame: {py_first_frame}, C++ first frame: {cpp_first_frame}")

    if py_first_frame == 0 and cpp_first_frame == 1:
        print("Notice: C++ frames are 1-indexed. Offsetting C++ frame by -1 to match Python 0-indexed.")
        cpp_df['frame_matched'] = cpp_df['frame'] - 1
    else:
        cpp_df['frame_matched'] = cpp_df['frame']

    merged = pd.merge(py_df, cpp_df, left_on='frame', right_on='frame_matched', suffixes=('_py', '_cpp'))
    print(f"Matched common frames: {len(merged)}")

    # 1. 状态一致性对比
    status_match = (merged['status_py'] == merged['status_cpp'])
    match_rate = status_match.mean() * 100.0
    print(f"\n--- 1. 检测状态匹配率: {match_rate:.2f}% ({status_match.sum()}/{len(merged)}) ---")

    mismatch = merged[~status_match]
    if len(mismatch) > 0:
        print(f"Found {len(mismatch)} status mismatches. First 10:")
        for _, row in mismatch.head(10).iterrows():
            f_num = row['frame_py']
            print(f"  Frame {f_num}: Python='{row['status_py']}' vs C++='{row['status_cpp']}'")
    else:
        print("ALL status 100% MATCHED!")

    # 2. 对双方都为 'ok' 的帧，对比 x_front, x_rear, cabin_length
    both_ok = merged[(merged['status_py'] == 'ok') & (merged['status_cpp'] == 'ok')]
    print(f"\n--- 2. 坐标数值对比 (Both 'ok' frames: {len(both_ok)}) ---")

    x_front_col = 'x_front' if 'x_front' in both_ok.columns else 'x_low_py'
    x_rear_col = 'x_rear' if 'x_rear' in both_ok.columns else 'x_high_py'
    len_col = 'cabin_length' if 'cabin_length' in both_ok.columns else 'cabin_length_py'

    # 对齐大小X：x_low为小端，x_high为大端
    py_low = np.minimum(both_ok[x_front_col], both_ok[x_rear_col])
    py_high = np.maximum(both_ok[x_front_col], both_ok[x_rear_col])
    cpp_low = both_ok['x_low']
    cpp_high = both_ok['x_high']

    diff_x_low = (py_low - cpp_low).abs()
    diff_x_high = (py_high - cpp_high).abs()
    diff_len = (both_ok[len_col] - both_ok['cabin_length_m']).abs()

    print(f"x_low  (小X端) 最大偏差: {diff_x_low.max():.4f} m, 平均偏差: {diff_x_low.mean():.4f} m")
    print(f"x_high (大X端) 最大偏差: {diff_x_high.max():.4f} m, 平均偏差: {diff_x_high.mean():.4f} m")
    print(f"cabin_len(车长) 最大偏差: {diff_len.max():.4f} m, 平均偏差: {diff_len.mean():.4f} m")

    close_match = (diff_len < 0.05).mean() * 100.0
    print(f"车长偏差 < 5cm 的帧比例: {close_match:.2f}%")

    # 3. 典型帧验证 (Frame 114, Frame 465)
    for target_f in [114, 465]:
        r = merged[merged['frame_py'] == target_f]
        if not r.empty:
            row = r.iloc[0]
            print(f"\n[典型帧 Frame {target_f} 详细比对]:")
            print(f"  Python -> status: {row['status_py']}, x_low: {min(row[x_front_col], row[x_rear_col])}, x_high: {max(row[x_front_col], row[x_rear_col])}, len: {row[len_col]}")
            print(f"  C++    -> status: {row['status_cpp']}, x_low: {row['x_low']}, x_high: {row['x_high']}, len: {row['cabin_length_m']}")

if __name__ == '__main__':
    import sys
    py_csv = sys.argv[1] if len(sys.argv) > 1 else 'python_results.csv'
    cpp_csv = sys.argv[2] if len(sys.argv) > 2 else 'cpp_results.csv'
    compare(py_csv, cpp_csv)
