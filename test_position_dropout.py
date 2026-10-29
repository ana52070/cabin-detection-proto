# -*- coding: utf-8 -*-
"""位置缺点、限时补偿及运动回归，不连接设备。"""
import unittest
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from unittest.mock import patch
from pathlib import Path
from dataclasses import replace
import numpy as np
from cabin_detector import CabinDetector, CabinPositionResult
from cabin_tracking import CabinPositionTracker
from short_edge_detector import CoarseClusterResult, ShortEdgeDetector
from tm16_decoder import TM16Frame
from cabin_timeseries import run_cabin_timeseries_pipeline
import config


def truck(shift=0.):
    # 连续底板加两端立面，覆盖真实粗聚类与几何检查。
    xs = np.repeat(np.arange(0., 10.05, .025), 8)
    floor = np.column_stack((xs, np.tile(np.linspace(-1., 1., 8), len(xs)//8), np.full(len(xs), 2.)))
    walls = [np.column_stack((np.full(400, x), np.tile(np.linspace(-1., 1., 20), 20),
                              np.repeat(np.linspace(1., 5., 20), 20))) for x in (0., 10.)]
    cloud = np.vstack([floor]+walls)
    cloud[:, 0] += shift
    return np.column_stack((cloud, np.ones((len(cloud), 2))))


class PositionDropoutTests(unittest.TestCase):
    def test_position_filter_does_not_drift_after_a_stop(self):
        from cabin_kalman import PositionKalmanFilter, KalmanConfig
        kf = PositionKalmanFilter(KalmanConfig(**config.KALMAN_CONFIG))
        errors = []
        for i in range(70):
            x = 0. if i < 30 else min((i-30)*.2, 4.)
            kf.update(x, x+10., i*.1)
            if 50 <= i <= 55:
                errors.append(abs(kf.current_estimate()[0]-x))
        self.assertLess(max(errors), .25)

    def test_detector_enforces_reference_length_and_motion(self):
        detector = CabinDetector()
        reference = CabinPositionResult('ok', x_front=10., x_rear=0., cabin_length=10.)
        coarse = CoarseClusterResult(truck(), np.zeros(200), (4., 10.), 6., 'ok')
        result = detector.detect(truck(), reference=reference, max_position_step=1.,
                                 length_tolerance=.6, coarse_res=coarse)
        self.assertNotEqual(result.status, 'ok')

    def test_missing_board_cannot_replace_reference(self):
        tracker = CabinPositionTracker(CabinDetector())
        points = truck()
        for i in range(5):
            result = tracker.update(points, i*.1)
        self.assertEqual(result.status, 'ok')
        reference = tracker.reference
        time = tracker.reference_time
        result = tracker.update(points[points[:, 0] < 5.], .5)
        self.assertNotEqual(result.status, 'ok')
        self.assertEqual(tracker.reference, reference)
        self.assertEqual(tracker.reference_time, time)
        restored = tracker.update(points, .6)
        self.assertEqual(restored.status, 'ok')
        self.assertAlmostEqual(restored.cabin_length, reference.cabin_length)

    def test_interior_gap_can_recover_both_measured_boards(self):
        tracker = CabinPositionTracker(CabinDetector())
        points = truck()
        for i in range(5):
            original = tracker.update(points, i*.1)
        damaged = points[(points[:, 0] < 2.) | (points[:, 0] > 7.)]
        result = tracker.update(damaged, .5, frame_complete=False)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.track_event, 'local_reacquired')
        self.assertLess(abs(result.cabin_length-original.cabin_length), .6)

    def test_normal_translation_is_retained(self):
        tracker = CabinPositionTracker(CabinDetector())
        results = [tracker.update(truck(i*.2), i*.1) for i in range(15)]
        self.assertEqual(results[-1].status, 'ok')
        self.assertGreater(results[-1].x_front-results[4].x_front, 1.5)

    def test_pipeline_predictions_expire_without_learning_shortened_length(self):
        frames = []
        for i in range(13):
            cloud = truck()
            if i >= 5:
                cloud = cloud[cloud[:, 0] < 5.]
            frames.append(TM16Frame(i, 100.+i*.1, cloud, cloud, 75 if i < 5 else 40,
                                     'TEST', np.zeros(16)))
        result = run_cabin_timeseries_pipeline(frames, ShortEdgeDetector(), CabinDetector(),
                    kalman_config=dict(config.KALMAN_CONFIG, enabled=True))
        points = [p for p in result['points'] if not p.is_gap_fill]
        self.assertEqual([p.position_type for p in points[5:10]], ['predicted']*5)
        self.assertEqual(points[10].position_type, 'invalid')
        self.assertLess(abs(points[5].cabin_length-points[4].cabin_length), .3)
        self.assertTrue(all(r.status != 'ok' for r in result['positions'][5:]))

    def test_roi_clipped_vehicle_is_not_a_shorter_complete_vehicle(self):
        detector = CabinDetector()
        coarse = CoarseClusterResult(truck(), np.zeros(200), (10., 21.), 11., 'ok')
        result = detector.detect(truck(), coarse_res=coarse)
        self.assertNotEqual(result.status, 'ok')

    def test_recorded_missing_front_uses_prediction(self):
        path = Path(__file__).parent/'verification_position_dropout'/'dropout_clouds.npz'
        if not path.exists():
            self.skipTest('实际缺点样本未随源码分发')
        with np.load(path) as sample:
            clouds = [sample['f1770']]*5 + [sample['f1771'], sample['f1772']]
        frames = [TM16Frame(i, 100.+i*.1, a, a, 66 if i == 5 else 75, 'TEST', np.zeros(16))
                  for i,a in enumerate(clouds)]
        result = run_cabin_timeseries_pipeline(frames, ShortEdgeDetector(), CabinDetector(),
                                               kalman_config=dict(config.KALMAN_CONFIG,enabled=True))
        points = [p for p in result['points'] if not p.is_gap_fill]
        self.assertEqual(points[5].position_type, 'predicted')
        self.assertGreater(abs(points[5].x_high_raw-points[4].x_high_raw), 6.)
        self.assertLess(abs(points[5].cabin_length-points[4].cabin_length), .3)
        self.assertEqual(points[6].position_type, 'reliable_measured')

    def test_departure_expires_instead_of_reporting_shrinking_length(self):
        frames = []
        for i in range(75):
            cloud = truck(i*.2)
            frames.append(TM16Frame(i, 100.+i*.1, cloud, cloud, 75, 'TEST', np.zeros(16)))
        result = run_cabin_timeseries_pipeline(frames, ShortEdgeDetector(), CabinDetector(),
                                               kalman_config=dict(config.KALMAN_CONFIG,enabled=True))
        points = [p for p in result['points'] if not p.is_gap_fill]
        first = next(i for i,p in enumerate(points) if p.position_type == 'predicted')
        self.assertEqual([p.position_type for p in points[first:first+5]], ['predicted']*5)
        self.assertTrue(all(p.position_type == 'invalid' for p in points[first+5:]))
        self.assertLess(abs(points[first+4].cabin_length-points[first-1].cabin_length), .3)

    def test_gui_never_falls_back_to_raw_after_sequential_rejection(self):
        from PySide6.QtWidgets import QApplication
        from visualizer_gui import TM16Visualizer
        app = QApplication.instance() or QApplication([])
        win = TM16Visualizer(auto_load=False)
        cloud = truck()
        win.frames = [TM16Frame(0, 100., cloud, cloud, 40, 'TEST', np.zeros(16))]
        win.current_idx = 0
        for status in ('position_rejected', 'predicted'):
            win.timeseries_win.position_results = [CabinPositionResult(status, x_front=10., x_rear=0., cabin_length=10.)]
            with patch.object(win.cabin_detector, 'detect', side_effect=AssertionError('禁止回退粗检测')):
                win._update_current_frame_display()
            self.assertIn('0x00', win.lbl_plc_mask.text())
        win.close()


if __name__ == '__main__':
    unittest.main()
