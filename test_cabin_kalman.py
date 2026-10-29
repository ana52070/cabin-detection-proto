# -*- coding: utf-8 -*-
import unittest
import numpy as np

from cabin_kalman import KalmanConfig, PositionKalmanFilter, SingleEndHeightKalman, CabinHeightKalmanFilter
from cabin_timeseries import CabinTimeSeriesAnalyzer

class KalmanUnitTests(unittest.TestCase):
    def test_switchability_kalman_disabled(self):
        cfg = KalmanConfig(enabled=False)
        filter = CabinHeightKalmanFilter(cfg)
        zf, zr, diff, z_final = filter.process(6.5, 6.6, 1.0)
        self.assertIsNone(zf)
        self.assertIsNone(zr)
        self.assertIsNone(z_final)

    def test_height_kalman_update_and_coast(self):
        cfg = KalmanConfig(enabled=True, max_coast_frames=3, max_coast_time=0.5,
                           height_process_noise_q=0.001, height_measure_noise_r=0.03)
        kf = SingleEndHeightKalman(cfg)
        # Initial measurement
        z0 = kf.update(6.50, 1.0)
        self.assertAlmostEqual(z0, 6.50, places=2)

        # Update with small noise
        z1 = kf.update(6.52, 1.1)
        self.assertTrue(6.50 <= z1 <= 6.52)

        # Coasting across missing frames
        c1 = kf.coast(1.2)
        self.assertIsNotNone(c1)
        self.assertAlmostEqual(c1, z1, places=3)

        c2 = kf.coast(1.3)
        self.assertIsNotNone(c2)

        c3 = kf.coast(1.4)
        self.assertIsNotNone(c3)

        # Exceeds max coast frames (3) -> should expire
        c4 = kf.coast(1.5)
        self.assertIsNone(c4)

    def test_outlier_rejection_innovation_gate(self):
        cfg = KalmanConfig(enabled=True, height_gate_limit=0.25)
        kf = SingleEndHeightKalman(cfg)
        kf.update(6.80, 1.0)
        kf.update(6.81, 1.1)

        # Gross outlier: laser falls to cabin floor (5.20m, 1.6m drop)
        z_outlier = kf.update(5.20, 1.2)
        # Should reject outlier and maintain predicted height near 6.80m
        self.assertIsNotNone(z_outlier)
        self.assertTrue(6.75 <= z_outlier <= 6.85)

    def test_position_kalman_tracking_and_coast(self):
        cfg = KalmanConfig(enabled=True, max_coast_frames=5, max_coast_time=0.6)
        pos_kf = PositionKalmanFilter(cfg)
        # Low X = -10.0, High X = 10.0
        pos_kf.update(x_low=-10.0, x_high=10.0, timestamp=1.0)
        pos_kf.update(x_low=-9.9, x_high=10.1, timestamp=1.1)
        pos_kf.update(x_low=-9.8, x_high=10.2, timestamp=1.2)

        coasted = pos_kf.coast(1.3)
        self.assertIsNotNone(coasted)
        xl, xh, length = coasted
        self.assertAlmostEqual(length, 20.0, delta=0.2)
        # low X moving positive (-9.8 -> -9.7), high X moving positive (10.2 -> 10.3)
        self.assertTrue(xl < xh, "x_low must always be smaller than x_high")
        self.assertTrue(xh > 10.15)
        self.assertTrue(xl > -9.85)

    def test_time_reversal_resets_filter(self):
        """Time jump backwards must reset state, not blow up matrices."""
        cfg = KalmanConfig(enabled=True)
        kf = SingleEndHeightKalman(cfg)
        kf.update(6.80, 10.0)
        kf.update(6.81, 10.1)
        # Time jump back to 5.0
        z_rev = kf.update(6.50, 5.0)
        self.assertAlmostEqual(z_rev, 6.50, places=2)

    def test_target_switching_isolation(self):
        """Switching target epoch resets filter and prevents cross-target contamination."""
        cfg = KalmanConfig(enabled=True)
        h_filter = CabinHeightKalmanFilter(cfg)
        zl1, zh1, d1, zf1 = h_filter.process(z_low_raw=6.80, z_high_raw=6.90, timestamp=1.0)
        self.assertAlmostEqual(zf1, 6.85, places=2)

        # Target switch: reset called
        h_filter.reset()
        self.assertFalse(h_filter.kf_low.is_initialized)
        self.assertFalse(h_filter.kf_high.is_initialized)

        # New vehicle with different height
        zl2, zh2, d2, zf2 = h_filter.process(z_low_raw=5.20, z_high_raw=5.30, timestamp=2.0)
        self.assertAlmostEqual(zf2, 5.25, places=2)

    def test_single_end_dropout_coasts_independently(self):
        """When one end drops out, it coasts while the other end updates normally."""
        cfg = KalmanConfig(enabled=True, max_coast_frames=5, max_coast_time=0.6)
        h_filter = CabinHeightKalmanFilter(cfg)
        h_filter.process(z_low_raw=6.80, z_high_raw=7.00, timestamp=1.0)
        h_filter.process(z_low_raw=6.81, z_high_raw=7.01, timestamp=1.1)

        # Low end drops out (None), high end still measures 7.02
        zl, zh, diff, z_final = h_filter.process(z_low_raw=None, z_high_raw=7.02, timestamp=1.2)
        self.assertIsNotNone(zl)
        self.assertIsNotNone(zh)
        self.assertAlmostEqual(zl, 6.81, places=1)
        self.assertAlmostEqual(zh, 7.02, places=2)
        self.assertIsNotNone(z_final)


if __name__ == '__main__':
    unittest.main()
