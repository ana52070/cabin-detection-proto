# -*- coding: utf-8 -*-
"""断档恢复、候选预热和因果输出的回归验证。"""
import unittest

import numpy as np

from cabin_detector import CabinDetector
from cabin_tracking import CabinPositionTracker
from cabin_timeseries import CabinTimeSeriesAnalyzer, TimeSeriesComputeWorker
from short_edge_detector import ShortEdgeDetector
from tm16_decoder import TM16Frame
from test_cabin_tracking import boards


def run_worker(times, clouds):
    frames = [TM16Frame(i, t, p, p, 75, 'DEFAULT', np.zeros(16))
              for i, (t, p) in enumerate(zip(times, clouds))]
    worker = TimeSeriesComputeWorker(
        frames, ShortEdgeDetector(), CabinDetector(enable_z_uniformity=False),
        CabinTimeSeriesAnalyzer(max_time_gap=.3, preserve_short_gaps=True))
    result, errors = [], []
    worker.finished.connect(result.append)
    worker.failed.connect(errors.append)
    worker.run()
    if errors:
        raise AssertionError(errors)
    return result[0]


class ParallelWarmupTests(unittest.TestCase):
    def setUp(self):
        self.cloud = boards([(0., 100), (12., 100)])

    def test_fifth_real_sample_can_output_without_rewriting_first_two_frames(self):
        result = run_worker([i*.1 for i in range(8)], [self.cloud]*8)
        pts = result['points']
        self.assertTrue(all(p.z_final is None for p in pts[:4]))
        self.assertIsNotNone(pts[4].z_final)
        self.assertEqual(result['positions'][0].status, 'tracking_acquiring')

    def test_gap_recovery_outputs_on_fifth_post_gap_sample(self):
        times = [i*.1 for i in range(8)] + [3.+i*.1 for i in range(7)]
        result = run_worker(times, [self.cloud]*len(times))
        pts = result['points']
        self.assertTrue(all(p.z_final is None for p in pts[8:12]))
        self.assertIsNotNone(pts[12].z_final)

    def test_incompatible_tentative_samples_cannot_enter_confirmed_window(self):
        false = self.cloud.copy()
        false[:, 0] += 5.
        false[:, 2] -= .8
        clouds = [false] + [self.cloud]*7
        result = run_worker([i*.1 for i in range(len(clouds))], clouds)
        self.assertIsNone(result['points'][4].z_final)
        self.assertIsNotNone(result['points'][5].z_final)
        self.assertGreater(result['points'][5].z_final, 6.8)

    def test_appending_frames_never_backfills_acquiring_outputs(self):
        times = [i*.1 for i in range(8)]
        full = run_worker(times, [self.cloud]*8)['points']
        for n in range(1, 8):
            prefix = run_worker(times[:n], [self.cloud]*n)['points']
            self.assertEqual([p.z_final for p in prefix], [p.z_final for p in full[:n]])

    def test_missing_frame_discards_unconfirmed_samples(self):
        empty = np.empty((0, 5))
        clouds = [self.cloud, empty] + [self.cloud]*7
        result = run_worker([i*.1 for i in range(len(clouds))], clouds)
        self.assertIsNone(result['points'][5].z_final)
        self.assertIsNotNone(result['points'][6].z_final)

    def test_confirmation_seeds_are_unique_past_measurements(self):
        result = run_worker([i*.1 for i in range(7)], [self.cloud]*7)
        rows = result['frames_data']
        self.assertEqual([s['frame_idx'] for s in rows[2]['warmup_samples']], [0, 1])
        self.assertTrue(all(not r['warmup_samples'] for r in rows[3:]))
        self.assertIsNone(result['points'][0].z_front_raw)
        self.assertIsNone(result['points'][1].z_final)

    def test_invalid_current_measurement_does_not_output_from_valid_seeds(self):
        data = [dict(frame_idx=2, t_rel=.2, status='ok', track_event='reacquired',
                     z_front_raw=None, z_rear_raw=None,
                     warmup_samples=[dict(frame_idx=i, t_rel=i*.1,
                                          z_front_raw=7., z_rear_raw=7.) for i in range(2)])]
        self.assertIsNone(CabinTimeSeriesAnalyzer().analyze_frames(data)['points'][0].z_final)

    def test_future_and_duplicate_seed_samples_are_rejected(self):
        for index, timestamp in [(2, .2), (3, .3), (1, .1)]:
            seeds = [dict(frame_idx=1, t_rel=.1, z_front_raw=7., z_rear_raw=7.),
                     dict(frame_idx=index, t_rel=timestamp, z_front_raw=7., z_rear_raw=7.)]
            row = dict(frame_idx=2, t_rel=.2, status='ok', track_event='reacquired',
                       z_front_raw=7., z_rear_raw=7., warmup_samples=seeds)
            with self.subTest(index=index), self.assertRaises(ValueError):
                CabinTimeSeriesAnalyzer().analyze_frames([row])

    def test_recovery_prefix_is_causal(self):
        times = [i*.1 for i in range(7)] + [2.8+i*.1 for i in range(7)]
        full = run_worker(times, [self.cloud]*len(times))['points']
        for n in range(7, len(times)):
            prefix = run_worker(times[:n], [self.cloud]*n)['points']
            self.assertEqual([p.z_final for p in prefix], [p.z_final for p in full[:n]])


class RecoveryIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tracker = CabinPositionTracker(CabinDetector(enable_z_uniformity=False))
        self.cloud = boards([(0., 100), (12., 100)])
        for i in range(6):
            self.locked = self.tracker.update(self.cloud, i*.1)

    def test_same_structure_after_short_outage_keeps_identity_epoch(self):
        result = None
        for t in (2.6, 2.7, 2.8):
            result = self.tracker.update(self.cloud, t)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.track_epoch, self.locked.track_epoch)

    def test_different_wall_profile_starts_new_identity(self):
        cloud = self.cloud.copy()
        cloud[:, 2] -= .8
        for t in (2.6, 2.7, 2.8):
            result = self.tracker.update(cloud, t)
        self.assertGreater(result.track_epoch, self.locked.track_epoch)

    def test_implausible_motion_starts_new_identity(self):
        cloud = self.cloud.copy()
        cloud[:, 0] += 10.
        for t in (2.6, 2.7, 2.8):
            result = self.tracker.update(cloud, t)
        self.assertGreater(result.track_epoch, self.locked.track_epoch)

    def test_long_outage_expires_reference(self):
        for t in (10., 10.1, 10.2):
            result = self.tracker.update(self.cloud, t)
        self.assertGreater(result.track_epoch, self.locked.track_epoch)

    def test_same_top_but_changed_wall_span_starts_new_identity(self):
        cloud = self.cloud.copy()
        cloud[:, 2] = 7.2 - (7.2-cloud[:, 2])*.7
        for t in (2.6, 2.7, 2.8):
            result = self.tracker.update(cloud, t)
        self.assertGreater(result.track_epoch, self.locked.track_epoch)

    def test_shared_translation_uses_recent_velocity_for_identity_recovery(self):
        tracker = CabinPositionTracker(CabinDetector(enable_z_uniformity=False))
        for i in range(6):
            cloud = self.cloud.copy()
            cloud[:, 0] += i*.1
            locked = tracker.update(cloud, i*.1)
        for t in (2.6, 2.7, 2.8):
            cloud = self.cloud.copy()
            cloud[:, 0] += t
            result = tracker.update(cloud, t)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.track_epoch, locked.track_epoch)
        self.assertEqual(result.track_event, 'resumed')


if __name__ == '__main__':
    unittest.main()
