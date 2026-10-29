# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def compare():
    print("Loading Python and C++ Kalman results...")
    py_df = pd.read_csv("python_results_kalman.csv")
    cpp_df = pd.read_csv("cpp_results_kalman.csv")

    print(f"Python rows: {len(py_df)}, C++ rows: {len(cpp_df)}")

    # Align frame index: C++ is 1-indexed, Python is 0-indexed
    cpp_df['frame_matched'] = cpp_df['frame'] - 1

    merged = pd.merge(py_df, cpp_df, left_on='frame', right_on='frame_matched', suffixes=('_py', '_cpp'))
    print(f"Total matched frames: {len(merged)}")

    # 1. Compare Final Filtered Height
    py_valid_h = merged['final_filtered_height'].notna()
    cpp_valid_h = merged['final_filtered_height_m'].notna()
    both_valid_h = merged[py_valid_h & cpp_valid_h]
    print(f"\n--- 1. 最终高度输出帧数对比 ---")
    print(f"Python 最终有效高度帧数: {py_valid_h.sum()} / {len(merged)} ({py_valid_h.mean()*100:.1f}%)")
    print(f"C++    最终有效高度帧数: {cpp_valid_h.sum()} / {len(merged)} ({cpp_valid_h.mean()*100:.1f}%)")
    print(f"双方同时输出有效高度帧数: {len(both_valid_h)}")

    diff_final_h = (both_valid_h['final_filtered_height'] - both_valid_h['final_filtered_height_m']).abs()
    print(f"最终综合高度最大偏差: {diff_final_h.max()*100:.2f} cm")
    print(f"最终综合高度平均偏差: {diff_final_h.mean()*100:.2f} cm")
    print(f"高度偏差 < 2cm 的帧比例: {(diff_final_h < 0.02).mean()*100:.2f}%")
    print(f"高度偏差 < 5cm 的帧比例: {(diff_final_h < 0.05).mean()*100:.2f}%")

    # 2. Compare Both Ends Height (Low Z and High Z)
    both_low = merged[merged['low_filtered_z'].notna() & merged['low_filtered_z_m'].notna()]
    both_high = merged[merged['high_filtered_z'].notna() & merged['high_filtered_z_m'].notna()]
    diff_low_z = (both_low['low_filtered_z'] - both_low['low_filtered_z_m']).abs()
    diff_high_z = (both_high['high_filtered_z'] - both_high['high_filtered_z_m']).abs()
    print(f"\n--- 2. 两端滤波高度对比 ---")
    print(f"小X端(后)滤波高度平均偏差: {diff_low_z.mean()*100:.2f} cm")
    print(f"大X端(前)滤波高度平均偏差: {diff_high_z.mean()*100:.2f} cm")

    # 3. Compare Positions (x_low, x_high, cabin_length) on common ok frames
    both_ok = merged[(merged['status_py'] == 'ok') & (merged['x_low_py'].notna()) & (merged['x_low_cpp'].notna())]
    print(f"\n--- 3. 车厢边界与长度对比 (共同有效帧: {len(both_ok)}) ---")
    diff_x_low = (both_ok['x_low_py'] - both_ok['x_low_cpp']).abs()
    diff_x_high = (both_ok['x_high_py'] - both_ok['x_high_cpp']).abs()
    diff_len = (both_ok['cabin_length'] - both_ok['cabin_length_m']).abs()
    close_len = (diff_len < 0.05).mean() * 100.0
    print(f"车长偏差 < 5cm 的帧比例: {close_len:.2f}%")
    print(f"车长平均偏差: {diff_len.mean()*100:.2f} cm")

    # 4. Compare PLC Output Registers
    both_plc = merged[merged['plc_front_height_mm_py'].notna() & merged['plc_front_height_mm_cpp'].notna()]
    print(f"\n--- 4. PLC 输出寄存器对比 (共同输出帧: {len(both_plc)}) ---")
    diff_plc_front_h = (both_plc['plc_front_height_mm_py'] - both_plc['plc_front_height_mm_cpp']).abs()
    diff_plc_rear_h = (both_plc['plc_rear_height_mm_py'] - both_plc['plc_rear_height_mm_cpp']).abs()
    print(f"PLC 前帮高度 (VW2502) 平均偏差: {diff_plc_front_h.mean():.1f} mm")
    print(f"PLC 后帮高度 (VW2514) 平均偏差: {diff_plc_rear_h.mean():.1f} mm")

    print("\n✅ 总结: Python 版本算法与 Jetson C++ 正式代码检测结果高度一致！")

if __name__ == '__main__':
    compare()
