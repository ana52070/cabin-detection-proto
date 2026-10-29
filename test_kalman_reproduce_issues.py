# -*- coding: utf-8 -*-
"""
Tests reproducing the identified Kalman filter issues in TM16 Python prototype.
These tests are written before fixing the bugs to confirm failures.
"""
import unittest
import numpy as np

from cabin_kalman import KalmanConfig, PositionKalmanFilter, SingleEndHeightKalman, CabinHeightKalmanFilter
from cabin_timeseries import CabinTimeSeriesAnalyzer
import config


class KalmanIssueReproductionTests(unittest.TestCase):

    def test_issue1_position_kalman_endpoint_order(self):
        """
        Issue 1:
        Detector convention is x_front = x_low (smaller X), x_rear = x_high (larger X).
        PositionKalmanFilter.predict() and coast() must return (x_low, x_high, length)
        such that result[0] < result[1].
        Currently, it returns (xc + 0.5*L, xc - 0.5*L) where result[0] > result[1],
        causing complete inversion of endpoints during coasting!
        """
        cfg = KalmanConfig(enabled=True, max_coast_frames=5, max_coast_time=0.6)
        pos_kf = PositionKalmanFilter(cfg)
        # Update with x_low=-5.0, x_high=16.0 (length=21.0, xc=5.5)
        pos_kf.update(x_low=-5.0, x_high=16.0, timestamp=1.0)
        
        # Test predict
        pred = pos_kf.predict(1.1)
        x_low_pred, x_high_pred, len_pred = pred[0], pred[1], pred[2]
        self.assertLess(x_low_pred, x_high_pred, 
                        f"Expected x_low < x_high, but got x0={x_low_pred}, x1={x_high_pred} (inverted!)")
        
        # Test coast
        coasted = pos_kf.coast(1.2)
        self.assertIsNotNone(coasted)
        x_low_c, x_high_c, len_c = coasted[0], coasted[1], coasted[2]
        self.assertLess(x_low_c, x_high_c,
                        f"Expected coasted x_low < x_high, but got x0={x_low_c}, x1={x_high_c} (inverted!)")

    def test_issue1_height_filter_never_swaps_low_and_high(self):
        """
        Height filter should accept and return heights corresponding to x_low and x_high.
        Under no circumstances should the height of low end jump to high end or vice-versa.
        """
        cfg = KalmanConfig(enabled=True, max_coast_frames=5, max_coast_time=0.6)
        filter = CabinHeightKalmanFilter(cfg)
        # Low end = 6.80m, High end = 7.10m
        zl1, zh1, diff1, zfinal1 = filter.process(6.80, 7.10, 1.0)
        self.assertAlmostEqual(zl1, 6.80, places=2)
        self.assertAlmostEqual(zh1, 7.10, places=2)
        
        # In next frame, coast low end while updating high end
        zl2, zh2, diff2, zfinal2 = filter.process(None, 7.11, 1.1)
        self.assertAlmostEqual(zl2, 6.80, places=1)
        self.assertAlmostEqual(zh2, 7.10, places=1)
        self.assertTrue(zl2 < zh2, "Low end height should remain distinct from high end height")

    def test_issue4_inconsistent_outliers_should_not_reset_state(self):
        """
        Issue 4:
        Currently, SingleEndHeightKalman simply counts outlier_count >= 3
        and blindly accepts whatever the 3rd outlier was!
        If input is 6.8m, then 5.2m, 6.0m, 4.5m (wildly inconsistent noise),
        it resets 6.8m to 4.5m!
        A true height step must be confirmed by consecutive MUTUALLY CONSISTENT observations.
        """
        cfg = KalmanConfig(enabled=True, height_gate_limit=0.25)
        kf = SingleEndHeightKalman(cfg)
        kf.update(6.80, 1.0)
        kf.update(6.80, 1.1)
        
        # Feed 3 wildly divergent outliers
        z1 = kf.update(5.20, 1.2)
        z2 = kf.update(6.00, 1.3)
        z3 = kf.update(4.50, 1.4)
        
        # State should NOT jump to 4.50m! It should either remain near 6.80m (coasted)
        # or expire/reject because the outliers are not mutually consistent.
        self.assertFalse(4.40 <= kf.z <= 4.60,
                         f"Kalman state wrongly reset to arbitrary inconsistent outlier: {kf.z}")

    def test_genuine_step_transition_when_mutually_consistent(self):
        """
        When consecutive observations are mutually consistent (e.g. 6.20, 6.21, 6.20),
        it should confirm a genuine step change from 6.80m to ~6.20m.
        """
        cfg = KalmanConfig(enabled=True, height_gate_limit=0.25)
        kf = SingleEndHeightKalman(cfg)
        kf.update(6.80, 1.0)
        kf.update(6.80, 1.1)
        
        # 3 consistent observations of a genuine step
        kf.update(6.20, 1.2)
        kf.update(6.21, 1.3)
        z3 = kf.update(6.20, 1.4)
        
        self.assertTrue(6.15 <= z3 <= 6.25,
                        f"Expected transition to ~6.20m after 3 consistent samples, got {z3}")

    def test_issue2_coast_timeout_behavior(self):
        """
        Issue 2:
        Coasting must expire when elapsed time > max_coast_time OR frames > max_coast_frames.
        """
        cfg = KalmanConfig(enabled=True, max_coast_frames=5, max_coast_time=0.60)
        kf = SingleEndHeightKalman(cfg)
        kf.update(6.80, 10.0)
        
        # Within coast time and frames
        c1 = kf.coast(10.1)
        self.assertIsNotNone(c1)
        c2 = kf.coast(10.2)
        self.assertIsNotNone(c2)
        
        # When 2.5s elapses in a single jump (like between frames 808 and 809)
        c_long = kf.coast(12.5) # 2.5s elapsed
        self.assertIsNone(c_long, "Coast should return None when elapsed time > 0.60s")

    def test_issue3_quality_counts_never_exceed_raw_candidates(self):
        """
        Issue 3:
        Quality pass count must NEVER exceed raw measured candidate count.
        Predictions must NOT be counted as 'pass_quality'.
        """
        frames_data = [
            {'frame_idx': 0, 't_rel': 0.0, 'status': 'ok', 'z_low_raw': 6.8, 'z_high_raw': 7.0, 'track_epoch': 1},
            {'frame_idx': 1, 't_rel': 0.1, 'status': 'no_valid_pair', 'z_low_raw': None, 'z_high_raw': None, 'track_epoch': 1},
            {'frame_idx': 2, 't_rel': 0.2, 'status': 'no_valid_pair', 'z_low_raw': None, 'z_high_raw': None, 'track_epoch': 1},
        ]
        analyzer = CabinTimeSeriesAnalyzer(kalman_config={'enabled': True, 'max_coast_frames': 5, 'max_coast_time': 0.6})
        res = analyzer.analyze_frames(frames_data)
        
        # Raw measurements: only frame 0 has raw measurements
        self.assertEqual(res.get('raw_low_count', res.get('raw_front_count')), 1)
        # Quality pass count MUST be <= 1 (not 3!)
        pass_low = res.get('pass_low_count', res.get('pass_front_count'))
        self.assertLessEqual(pass_low, 1, 
                             f"Quality pass count ({pass_low}) must not exceed raw candidate count (1)")


if __name__ == '__main__':
    unittest.main()
