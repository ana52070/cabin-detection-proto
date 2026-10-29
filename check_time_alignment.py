# -*- coding: utf-8 -*-
import pandas as pd

py = pd.read_csv('python_results.csv')
cpp = pd.read_csv('cpp_results.csv')

print("Python first 8 frames:")
for i in range(8):
    row = py.iloc[i]
    print(f"Py Frame {int(row['frame']):2d}: t={row['timestamp']:10.4f}, roi_pts={int(row['roi_points'])}")

print("\nC++ first 8 frames:")
for i in range(8):
    row = cpp.iloc[i]
    print(f"Cpp Frame {int(row['frame']):2d}: t_ms={row['time_ms']:10.4f}, roi_pts={int(row['roi_points'])}")
