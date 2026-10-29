# -*- coding: utf-8 -*-
"""端点身份与因果滤波回归测试，不连接雷达、PLC 或远端设备。"""
import unittest

import numpy as np

from cabin_detector import CabinDetector
from cabin_timeseries import CabinTimeSeriesAnalyzer
from cabin_tracking import CabinPositionTracker


def series(values):
    return [dict(frame_idx=i, t_rel=i * 0.1,
                 z_front_raw=z, z_rear_raw=z) for i, z in enumerate(values)]


def boards(spec):
    rng = np.random.default_rng(42)
    parts = []
    for x, count in spec:
        p = np.zeros((count, 5))
        p[:, 0] = x + rng.uniform(-0.025, 0.025, count)
        p[:, 1] = rng.uniform(-1.5, 1.5, count)
        p[:, 2] = rng.uniform(4.2, 7.2, count)
        parts.append(p)
    return np.concatenate(parts)


class CausalMedianTests(unittest.TestCase):
    def test_requires_five_current_and_history_samples(self):
        pts = CabinTimeSeriesAnalyzer().analyze_frames(series([7.] * 8))['points']
        self.assertEqual([p.z_final for p in pts[:4]], [None] * 4)
        self.assertEqual(pts[4].z_final, 7.)

    def test_future_cannot_change_past(self):
        analyzer = CabinTimeSeriesAnalyzer()
        data = series([7.] * 7 + [6.] * 6)
        full = analyzer.analyze_frames(data)['points']
        for n in range(1, len(data) + 1):
            prefix = analyzer.analyze_frames(data[:n])['points']
            self.assertEqual([p.z_final for p in prefix],
                             [p.z_final for p in full[:n]])

    def test_missing_frame_restarts_warmup(self):
        pts = CabinTimeSeriesAnalyzer().analyze_frames(
            series([7.] * 6 + [None] + [7.] * 6))['points']
        self.assertTrue(all(p.z_final is None for p in pts[6:11]))
        self.assertEqual(pts[11].z_final, 7.)

    def test_time_gap_restarts_warmup(self):
        data = series([7.] * 11)
        for row in data[6:]:
            row['t_rel'] += 2.
        pts = CabinTimeSeriesAnalyzer().analyze_frames(data)['points']
        self.assertTrue(all(p.z_final is None for p in pts[6:10]))
        self.assertEqual(pts[10].z_final, 7.)

    def test_mixed_surfaces_do_not_produce_intermediate_height(self):
        pts = CabinTimeSeriesAnalyzer().analyze_frames(
            series([7.2] * 6 + [5.4] * 5))['points']
        self.assertTrue(all(p.z_final is None for p in pts[6:10]))
        self.assertEqual(pts[10].z_final, 5.4)

    def test_one_bad_end_does_not_reset_other_end(self):
        data = series([7.] * 12)
        data[6]['z_rear_raw'] = float('nan')
        pts = CabinTimeSeriesAnalyzer().analyze_frames(data)['points']
        self.assertEqual(pts[7].z_front_med, 7.)
        self.assertIsNone(pts[7].z_rear_med)

    def test_rejected_detection_is_not_a_height_measurement(self):
        data = series([7.] * 10)
        data[6]['status'] = 'tracking_rejected'
        pts = CabinTimeSeriesAnalyzer().analyze_frames(data)['points']
        self.assertIsNone(pts[6].z_final)

    def test_slow_ramp_checks_only_current_window(self):
        values = [6. + i * .045 for i in range(12)]
        pts = CabinTimeSeriesAnalyzer().analyze_frames(series(values))['points']
        self.assertTrue(all(p.z_final is not None for p in pts[4:]))
        self.assertAlmostEqual(pts[-1].z_final, values[-3])

    def test_short_gap_mode_never_fills_missing_but_can_resume_current_measurement(self):
        analyzer = CabinTimeSeriesAnalyzer(preserve_short_gaps=True, max_history_age=.7)
        pts = analyzer.analyze_frames(series([7.] * 6 + [None] + [7.] * 4))['points']
        self.assertIsNone(pts[6].z_final)
        self.assertEqual(pts[7].z_final, 7.)

    def test_short_gap_mode_rejects_current_spike_without_poisoning_history(self):
        analyzer = CabinTimeSeriesAnalyzer(preserve_short_gaps=True, max_history_age=.7)
        pts = analyzer.analyze_frames(series([7.] * 6 + [5.4] + [7.] * 4))['points']
        self.assertIsNone(pts[6].z_final)
        self.assertEqual(pts[7].z_final, 7.)

    def test_short_gap_mode_accepts_confirmed_height_change(self):
        analyzer = CabinTimeSeriesAnalyzer(preserve_short_gaps=True, max_history_age=.7)
        pts = analyzer.analyze_frames(series([7.2] * 6 + [5.4] * 6))['points']
        self.assertTrue(all(p.z_final is None for p in pts[6:10]))
        self.assertEqual(pts[10].z_final, 5.4)

    def test_short_gap_mode_expires_history(self):
        analyzer = CabinTimeSeriesAnalyzer(preserve_short_gaps=True, max_history_age=.7)
        pts = analyzer.analyze_frames(series([7.] * 6 + [None] * 8 + [7.] * 5))['points']
        self.assertTrue(all(p.z_final is None for p in pts[14:18]))
        self.assertEqual(pts[18].z_final, 7.)

    def test_identity_change_always_clears_height_history(self):
        data = series([7.] * 12)
        for row in data[6:]:
            row['track_epoch'] = 1
        pts = CabinTimeSeriesAnalyzer().analyze_frames(data)['points']
        self.assertTrue(all(p.z_final is None for p in pts[6:10]))

    def test_short_gap_mode_is_causal(self):
        analyzer = CabinTimeSeriesAnalyzer(preserve_short_gaps=True, max_history_age=.7)
        data = series([7.] * 6 + [None, 5.4, 7., 7., 7., 7.])
        full = analyzer.analyze_frames(data)['points']
        for n in range(1, len(data)):
            prefix = analyzer.analyze_frames(data[:n])['points']
            self.assertEqual([p.z_final for p in prefix], [p.z_final for p in full[:n]])

    def test_invalid_filter_parameters_are_rejected(self):
        for kwargs in (dict(median_window=4), dict(max_history_age=0),
                       dict(max_height_spread=float('nan')), dict(max_time_gap=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CabinTimeSeriesAnalyzer(**kwargs)


class CandidateIdentityTests(unittest.TestCase):
    def setUp(self):
        self.detector = CabinDetector(enable_z_uniformity=False)
        self.reference = self.detector.detect(boards([(0., 100), (12., 100)]))
        self.assertEqual(self.reference.status, 'ok')

    def test_stronger_interior_peak_cannot_replace_tracked_board(self):
        result = self.detector.detect(
            boards([(0.1, 100), (6., 400), (12.1, 100)]),
            reference=self.reference, max_position_step=0.8,
            length_tolerance=0.6)
        self.assertEqual(result.status, 'ok')
        self.assertAlmostEqual(result.x_front, 0.1, delta=0.2)
        self.assertAlmostEqual(result.x_rear, 12.1, delta=0.2)

    def test_no_compatible_pair_returns_invalid_instead_of_old_position(self):
        result = self.detector.detect(
            boards([(6., 100), (18., 100)]),
            reference=self.reference, max_position_step=0.8,
            length_tolerance=0.6)
        self.assertEqual(result.status, 'tracking_rejected')
        self.assertIsNone(result.x_front)


class SequentialTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = CabinPositionTracker(CabinDetector(enable_z_uniformity=False))
        self.cloud = boards([(0., 100), (12., 100)])

    def lock(self):
        results = [self.tracker.update(self.cloud, i * .1) for i in range(3)]
        self.assertEqual([r.status for r in results], ['tracking_acquiring'] * 2 + ['ok'])

    def test_short_false_candidate_does_not_replace_reference(self):
        self.lock()
        false_cloud = boards([(6., 100), (18., 100)])
        self.assertEqual(self.tracker.update(false_cloud, .3).status, 'tracking_acquiring')
        result = self.tracker.update(self.cloud, .4)
        self.assertEqual(result.status, 'ok')
        self.assertAlmostEqual(result.x_front, 0., delta=.2)

    def test_missing_never_returns_held_position(self):
        self.lock()
        result = self.tracker.update(np.empty((0, 5)), .3)
        self.assertEqual(result.status, 'no_points')
        self.assertIsNone(result.x_front)

    def test_persistent_new_pair_reacquires(self):
        self.lock()
        cloud = boards([(6., 100), (18., 100)])
        results = [self.tracker.update(cloud, .3 + i * .1) for i in range(3)]
        self.assertEqual([r.status for r in results], ['tracking_acquiring'] * 2 + ['ok'])
        self.assertAlmostEqual(results[-1].x_front, 6., delta=.2)

    def test_persistent_stronger_pair_can_correct_initial_wrong_lock(self):
        self.lock()
        # 旧目标仍然可见，但另一个更强的真实端点持续出现。
        cloud = boards([(0., 100), (12., 100), (20., 400)])
        results = [self.tracker.update(cloud, .3 + i * .1) for i in range(3)]
        self.assertAlmostEqual(results[-1].x_rear, 20., delta=.2)

    def test_timestamp_gap_restarts_acquisition(self):
        self.lock()
        result = self.tracker.update(self.cloud, 2.)
        self.assertEqual(result.status, 'tracking_acquiring')

    def test_reversed_timestamp_restarts_acquisition(self):
        self.lock()
        self.assertEqual(self.tracker.update(self.cloud, .1).status, 'tracking_acquiring')

    def test_short_stronger_peak_with_old_pair_visible_does_not_switch(self):
        self.lock()
        cloud = boards([(0., 100), (6., 400), (12., 100)])
        for t in (.3, .4):
            result = self.tracker.update(cloud, t)
            self.assertAlmostEqual(result.x_front, 0., delta=.2)
            self.assertAlmostEqual(result.x_rear, 12., delta=.2)
        self.assertEqual(self.tracker.update(self.cloud, .5).status, 'ok')

    def test_entire_tracking_sequence_is_causal(self):
        clouds = [self.cloud] * 6 + [boards([(6., 100), (18., 100)])] * 4
        def run(items):
            tracker = CabinPositionTracker(CabinDetector(enable_z_uniformity=False))
            return [(r.status, r.x_front, r.x_rear)
                    for i, cloud in enumerate(items)
                    for r in [tracker.update(cloud, i * .1)]]
        full = run(clouds)
        for n in range(1, len(clouds)):
            self.assertEqual(run(clouds[:n]), full[:n])


class WorkerIntegrationTests(unittest.TestCase):
    def test_parameter_snapshots_and_sequential_output(self):
        from cabin_timeseries import TimeSeriesComputeWorker
        from short_edge_detector import ShortEdgeDetector
        from tm16_decoder import TM16Frame
        cloud = boards([(0., 100), (12., 100)])
        frames = [TM16Frame(i, i * .1, cloud, cloud, 75, 'DEFAULT', np.zeros(16))
                  for i in range(12)]
        roi = ShortEdgeDetector()
        detector = CabinDetector(enable_z_uniformity=False)
        worker = TimeSeriesComputeWorker(frames, roi, detector, CabinTimeSeriesAnalyzer())
        detector.min_peak_pts = 10000
        roi.roi['x_max'] = -100.
        result, errors = [], []
        worker.finished.connect(result.append)
        worker.failed.connect(errors.append)
        worker.run()
        self.assertFalse(errors)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]['positions']), 12)
        pts = result[0]['points']
        self.assertTrue(all(p.z_final is None for p in pts[:4]))
        self.assertIsNotNone(pts[4].z_final)


if __name__ == '__main__':
    unittest.main()
