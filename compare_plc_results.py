# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def main():
    py = pd.read_csv('python_results.csv')
    cpp = pd.read_csv('cpp_results.csv')

    print(f"Loaded Python: {len(py)} rows, C++: {len(cpp)} rows")

    # 对比前 686 帧
    N = 686
    py_sub = py.iloc[:N]
    cpp_sub = cpp.iloc[:N]

    # 筛选双方都判定为 ok 的帧
    valid_mask = (py_sub['status'] == 'ok') & (cpp_sub['status'] == 'ok')
    py_ok = py_sub[valid_mask]
    cpp_ok = cpp_sub[valid_mask]

    print(f"\nEvaluating PLC 4-field output on {len(py_ok)} frames (Radar installation offset = 1200 mm):")

    # 1. 车头位置 (Reg 43): 1200 + 大X * 1000
    py_head = py_ok['plc_head_position_mm'].astype(float)
    cpp_head = cpp_ok['plc_head_position_mm'].astype(float)
    head_diff = (py_head - cpp_head).abs()
    print(f"车头位置 (Reg 43) 最大偏差: {head_diff.max():.2f} mm, 平均偏差: {head_diff.mean():.2f} mm")
    print(f"车头位置 100% 精确一致率 (diff == 0 mm): {(head_diff == 0).mean() * 100:.2f}%")

    # 2. 车尾位置 (Reg 44): 1200 + 小X * 1000
    py_tail = py_ok['plc_tail_position_mm'].astype(float)
    cpp_tail = cpp_ok['plc_tail_position_mm'].astype(float)
    tail_diff = (py_tail - cpp_tail).abs()
    print(f"车尾位置 (Reg 44) 最大偏差: {tail_diff.max():.2f} mm, 平均偏差: {tail_diff.mean():.2f} mm")
    print(f"车尾位置 100% 精确一致率 (diff == 0 mm): {(tail_diff == 0).mean() * 100:.2f}%")

    # 3. 典型帧 Frame 465 详细展示
    f465_py = py_sub[py_sub['frame'] == 465].iloc[0]
    f465_cpp = cpp_sub[cpp_sub['frame'] == 466].iloc[0]

    print(f"\n[典型帧 Frame 465 PLC 字段对比 (21.76m 长挂车)]:")
    print(f"  车头位置 (Reg 43):")
    print(f"    Python: {f465_py['plc_head_position_mm']} mm (计算公式: 1200 + 7.6519 * 1000)")
    print(f"    C++   : {f465_cpp['plc_head_position_mm']} mm (计算公式: 1200 + 7.6519 * 1000)")
    print(f"  车尾位置 (Reg 44):")
    print(f"    Python: {f465_py['plc_tail_position_mm']} mm (计算公式: 1200 + (-14.1081) * 1000)")
    print(f"    C++   : {f465_cpp['plc_tail_position_mm']} mm (计算公式: 1200 + (-14.1081) * 1000)")
    print(f"  前帮高度 (Reg  1):")
    print(f"    Python: {f465_py['plc_front_height_mm']} mm")
    print(f"    C++   : {f465_cpp['plc_front_height_mm']} mm")
    print(f"  后帮高度 (Reg  7):")
    print(f"    Python: {f465_py['plc_rear_height_mm']} mm")
    print(f"    C++   : {f465_cpp['plc_rear_height_mm']} mm")

if __name__ == '__main__':
    main()
