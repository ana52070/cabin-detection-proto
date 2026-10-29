# -*- coding: utf-8 -*-
"""目标区间修复的端点语义、缺测和界面回归，不连接设备。"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from cabin_timeseries import CabinTimeSeriesAnalyzer, CabinTimeSeriesWindow, check_local_board_geometry
from cabin_measurement import measure_continuous_board_height
import config


class EndpointTests(unittest.TestCase):
    def test_grid_boundary_does_not_cut_off_recorded_board_top(self):
        path = Path(__file__).parent / 'verification_focus_1727_2294' / 'sample_clouds.npz'
        if not path.exists():
            self.skipTest('本地录包样本未随源码分发')
        with np.load(path) as samples:
            # 第1787和1788帧：粗网格边界变化一格，车帮顶部应保持一致。
            a = measure_continuous_board_height(samples['f1786'], -12.17,
                                                **config.BOARD_HEIGHT_CONFIG)[0]
            b = measure_continuous_board_height(samples['f1787'], -11.9025,
                                                **config.BOARD_HEIGHT_CONFIG)[0]
        self.assertLess(abs(a-b), .03)

    def test_explicit_endpoints_take_precedence_in_median_mode(self):
        rows = [dict(frame_idx=i, t_rel=i*.1, status='ok',
                     z_front_raw=6.7, z_rear_raw=6.4,
                     z_low_raw=6.4, z_high_raw=6.7) for i in range(8)]
        point = CabinTimeSeriesAnalyzer().analyze_frames(rows)['points'][-1]
        self.assertAlmostEqual(point.z_low_filtered, 6.4)
        self.assertAlmostEqual(point.z_high_filtered, 6.7)

    def test_local_geometry_uses_same_continuous_segment_as_measurement(self):
        # 底部连续车帮上方有悬空点，恢复不能改用整个切片的P98。
        z = np.r_[np.linspace(3., 6.6, 200), np.repeat(7.3, 20)]
        cloud = np.column_stack([np.zeros(len(z)), np.zeros(len(z)), z])
        self.assertLess(check_local_board_geometry(cloud, 0.), 6.7)

    def test_rejected_local_candidate_remains_prediction(self):
        rows = [dict(frame_idx=i, t_rel=i*.1, status='ok', track_epoch=1,
                     z_low_raw=6.5 if i == 0 else 5., z_high_raw=6.6,
                     is_local_reacquired_low=i > 0) for i in range(2)]
        p = CabinTimeSeriesAnalyzer(kalman_config=dict(config.KALMAN_CONFIG, enabled=True)).analyze_frames(rows)['points'][-1]
        self.assertEqual(p.point_type_low, 'predicted')
        self.assertFalse(p.pass_low_qual)


class PositionWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_three_position_plots_exist_and_invalidate(self):
        win = CabinTimeSeriesWindow()
        self.assertEqual(len(win.position_plots), 3)
        win.invalidate_results()
        self.assertTrue(all(curve.xData is None for curve in win.position_curves))
        win.close()

    def test_missing_time_does_not_draw_a_continuous_measured_line(self):
        from cabin_timeseries import TimeSeriesPoint
        win = CabinTimeSeriesWindow()
        points = [TimeSeriesPoint(i, t, z_front_med=6.5, point_type_low='reliable_measured')
                  for i, t in enumerate((0., .1, 2.))]
        win._on_worker_finished(dict(points=points))
        self.assertTrue(np.isnan(win.curve_low_meas.yData).any())
        win.close()

    def test_worker_copies_height_parameters(self):
        from cabin_timeseries import TimeSeriesComputeWorker
        from cabin_detector import CabinDetector
        from short_edge_detector import ShortEdgeDetector
        parameters = dict(config.BOARD_HEIGHT_CONFIG, x_margin=.8)
        worker = TimeSeriesComputeWorker([], ShortEdgeDetector(), CabinDetector(),
                                         CabinTimeSeriesAnalyzer(), parameters)
        parameters['x_margin'] = .1
        self.assertEqual(worker.height_config['x_margin'], .8)

    def test_position_cursor_uses_interface_frame_number(self):
        from cabin_timeseries import TimeSeriesPoint
        win = CabinTimeSeriesWindow()
        win._on_worker_finished(dict(points=[TimeSeriesPoint(1786, 178.6,
            x_low=-12., x_high=13., cabin_length=25., position_type='reliable_measured')]))
        win.set_current_frame(1786)
        text = win.lbl_cursor_info.text()
        self.assertIn('帧 #1787', text)
        self.assertIn('前X=13.000m 后X=-12.000m 长=25.000m', text)
        win.close()


if __name__ == '__main__':
    unittest.main()
