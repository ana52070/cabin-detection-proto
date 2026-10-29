# -*- coding: utf-8 -*-
import pandas as pd

py = pd.read_csv('python_results.csv')
cpp = pd.read_csv('cpp_results.csv')

print(f"Py len: {len(py)}, Cpp len: {len(cpp)}")

for i in range(len(cpp)):
    py_pts = py.iloc[i]['roi_points']
    cpp_pts = cpp.iloc[i]['roi_points']
    if py_pts != cpp_pts:
        print(f"Mismatch at index {i} (Py frame {py.iloc[i]['frame']}, Cpp frame {cpp.iloc[i]['frame']}): Py={py_pts} vs Cpp={cpp_pts}")
        # 看看是不是偏移了
        for off in range(-3, 4):
            if 0 <= i + off < len(py):
                print(f"  Py[{i+off}]={py.iloc[i+off]['roi_points']}")
        break
