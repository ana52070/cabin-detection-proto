# -*- coding: utf-8 -*-

with open('visualizer_gui.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. 导入 plc_adapter
if 'from plc_adapter import compute_plc_measurements' not in code:
    code = code.replace(
        'from cabin_timeseries import CabinTimeSeriesWindow',
        'from cabin_timeseries import CabinTimeSeriesWindow\nfrom plc_adapter import compute_plc_measurements, PlcMeasurements'
    )

# 2. UI 界面中增加 grp_plc
plc_ui_block = """
        # =====================================================================
        # Group 6.5: 🏭 临港 PLC 寄存器输出 (雷达安装基准 1200mm)
        # =====================================================================
        grp_plc = QGroupBox("🏭 临港 PLC 寄存器输出 (雷达基准 1200mm)")
        vbox_plc = QVBoxLayout(grp_plc)
        self.lbl_plc_head = QLabel("车头位置 (Reg 43): - mm (1200 + 大X×1000)")
        self.lbl_plc_tail = QLabel("车尾位置 (Reg 44): - mm (1200 + 小X×1000)")
        self.lbl_plc_front_h = QLabel("前帮高度 (Reg  1): - mm (大X端顶部Z)")
        self.lbl_plc_rear_h = QLabel("后帮高度 (Reg  7): - mm (小X端顶部Z)")
        self.lbl_plc_mask = QLabel("有效掩码 (Mask  ): 0 (全部无效)")

        for lbl in (self.lbl_plc_head, self.lbl_plc_tail, self.lbl_plc_front_h, self.lbl_plc_rear_h, self.lbl_plc_mask):
            lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: #81d4fa;")
            vbox_plc.addWidget(lbl)
        right_layout.addWidget(grp_plc)
"""

if 'grp_plc' not in code:
    target_ui = 'right_layout.addWidget(grp_cabin_diag)'
    code = code.replace(target_ui, target_ui + '\n' + plc_ui_block, 1)

# 3. 在 _update_current_frame_display 中更新 PLC 显示
plc_update_block = """
        # 2.5 Compute PLC 4-field measurements (with 1200mm installation base)
        zf_raw, zr_raw = None, None
        if cabin_res.status == "ok":
            b1 = roi_pts[(roi_pts[:, 0] >= cabin_res.x_front - 0.2) & (roi_pts[:, 0] <= cabin_res.x_front + 0.2)]
            if len(b1) > 0:
                zf_raw = float(np.percentile(b1[:, 2], 98))
            b2 = roi_pts[(roi_pts[:, 0] >= cabin_res.x_rear - 0.2) & (roi_pts[:, 0] <= cabin_res.x_rear + 0.2)]
            if len(b2) > 0:
                zr_raw = float(np.percentile(b2[:, 2], 98))

        plc_meas = compute_plc_measurements(
            cabin_res.x_front, cabin_res.x_rear, zf_raw, zr_raw, cabin_res.status
        )
        if plc_meas.is_head_valid and plc_meas.is_tail_valid:
            self.lbl_plc_head.setText(f"车头位置 (Reg 43): {plc_meas.head_position_mm:6d} mm")
            self.lbl_plc_tail.setText(f"车尾位置 (Reg 44): {plc_meas.tail_position_mm:6d} mm")
            fh_str = f"{plc_meas.front_height_mm:5d} mm" if plc_meas.is_front_h_valid else "缺测"
            rh_str = f"{plc_meas.rear_height_mm:5d} mm" if plc_meas.is_rear_h_valid else "缺测"
            self.lbl_plc_front_h.setText(f"前帮高度 (Reg  1): {fh_str}")
            self.lbl_plc_rear_h.setText(f"后帮高度 (Reg  7): {rh_str}")
            self.lbl_plc_mask.setText(f"有效掩码 (Mask  ): 0x{plc_meas.valid_mask:02X} (有效位: {bin(plc_meas.valid_mask)})")
        else:
            self.lbl_plc_head.setText("车头位置 (Reg 43): 0 mm (缺测)")
            self.lbl_plc_tail.setText("车尾位置 (Reg 44): 0 mm (缺测)")
            self.lbl_plc_front_h.setText("前帮高度 (Reg  1): 0 mm (缺测)")
            self.lbl_plc_rear_h.setText("后帮高度 (Reg  7): 0 mm (缺测)")
            self.lbl_plc_mask.setText("有效掩码 (Mask  ): 0 (全部无效)")
"""

fn_name = 'def _update_current_frame_display'
if fn_name in code and 'compute_plc_measurements' not in code.split(fn_name)[1]:
    idx = code.find(fn_name)
    sub = code[idx:]
    target_str = '# 3. Run Step 2 (Coarse clustering for comparison)'
    sub_new = sub.replace(target_str, plc_update_block + '\n        ' + target_str, 1)
    code = code[:idx] + sub_new

with open('visualizer_gui.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("visualizer_gui.py patched successfully!")
