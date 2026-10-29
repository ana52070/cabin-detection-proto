# -*- coding: utf-8 -*-
"""顺序帧端点跟踪与有限期身份恢复，不以历史位置冒充本帧测量。"""
import math
from collections import deque
from dataclasses import replace

import numpy as np

from cabin_detector import CabinDetector


class CabinPositionTracker:
    """运动、长度、立面联合关联；确认候选对高度预热可见，对正式输出不可见。"""

    def __init__(self, detector: CabinDetector, position_slack=0.5,
                 max_speed=5.0, length_tolerance=0.6,
                 reset_gap=1.0, confirm_frames=3, identity_memory=3.0,
                 recovery_acceleration=1.0, profile_tolerance=0.5):
        values = (position_slack, max_speed, length_tolerance, reset_gap,
                  identity_memory, recovery_acceleration, profile_tolerance)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("跟踪门限必须是有限正数")
        if identity_memory < reset_gap:
            raise ValueError("身份参考保留期不得小于重新确认门限")
        if not isinstance(confirm_frames, int) or confirm_frames < 2:
            raise ValueError("捕获确认帧数必须为不小于2的整数")
        self.detector = detector
        self.position_slack = position_slack
        self.max_speed = max_speed
        self.length_tolerance = length_tolerance
        self.reset_gap = reset_gap
        self.confirm_frames = confirm_frames
        self.identity_memory = identity_memory
        self.recovery_acceleration = recovery_acceleration
        self.profile_tolerance = profile_tolerance
        self.reset()

    def reset(self):
        self.epoch = getattr(self, 'epoch', 0)
        self._candidate_counter = getattr(self, '_candidate_counter', 0)
        self.reference = None
        self.reference_time = None
        self.reference_profile = None
        self.motion_history = deque(maxlen=6)
        self.length_history = deque(maxlen=31)
        self.raw_result = None
        self.last_time = None
        self.candidate = None
        self.candidate_id = 0
        self.confirmed_candidate_id = 0
        self._clear_pending()

    def _clear_pending(self):
        self.pending = None
        self.pending_time = None
        self.pending_profile = None
        self.pending_count = 0

    @staticmethod
    def _profile(points, result):
        signature = []
        for x in (result.x_front, result.x_rear):
            wall = points[np.abs(points[:, 0] - x) <= .65]
            if len(wall) < 5:
                return None
            lower, top = np.percentile(wall[:, 2], [5, 98])
            if top-lower < .9:
                return None
            signature.append((top, top - lower))
        return np.asarray(signature)

    def _profiles_match(self, a, b):
        return (a is not None and b is not None
                and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                and np.max(np.abs(a - b)) <= self.profile_tolerance)

    def _velocity(self):
        if len(self.motion_history) < 2:
            return 0.
        samples = np.asarray(self.motion_history)
        t = samples[:, 0] - samples[-1, 0]
        t -= t.mean()
        denom = float(t @ t)
        velocity = float(t @ samples[:, 1] / denom) if denom > 1e-9 else 0.
        return float(np.clip(velocity, -self.max_speed, self.max_speed))

    def _same_identity(self, result, profile, timestamp):
        if self.reference is None:
            return False
        dt = timestamp - self.reference_time
        if not 0 < dt <= self.identity_memory:
            return False
        shift = self._velocity() * dt
        allowance = self.position_slack + .5 * self.recovery_acceleration * dt * dt
        speed_limit = self.position_slack + self.max_speed * dt
        return (abs(result.cabin_length - self.reference.cabin_length) <= self.length_tolerance
                and abs(result.x_front - self.reference.x_front - shift) <= allowance
                and abs(result.x_rear - self.reference.x_rear - shift) <= allowance
                and abs(result.x_front - self.reference.x_front) <= speed_limit
                and abs(result.x_rear - self.reference.x_rear) <= speed_limit
                and self._profiles_match(self.reference_profile, profile))

    def _accept(self, result, profile, timestamp, event):
        result = replace(result, track_epoch=self.epoch, track_event=event)
        self.reference, self.reference_time = result, timestamp
        self.reference_profile = profile
        self.motion_history.append((timestamp, .5 * (result.x_front + result.x_rear)))
        self.length_history.append(result.cabin_length)
        return result

    def update(self, roi_points, timestamp, frame_complete=True):
        if not math.isfinite(timestamp):
            raise ValueError("帧时间必须有限")
        if self.last_time is not None and timestamp <= self.last_time:
            self.reset()
        if (self.reference_time is not None
                and timestamp - self.reference_time > self.identity_memory):
            self.reset()
        self.last_time = timestamp
        self.candidate = None
        self.confirmed_candidate_id = 0
        self.raw_result = self.detector.detect(roi_points)
        tracked = None
        dt = timestamp - self.reference_time if self.reference is not None else None
        if dt is not None and dt <= self.reset_gap:
            result = self.detector.detect(
                roi_points, reference=self.reference,
                max_position_step=self.position_slack + self.max_speed * dt,
                length_tolerance=self.length_tolerance)
            if (result.status == "ok" and frame_complete
                    and abs(result.cabin_length-np.median(self.length_history)) <= self.length_tolerance):
                tracked = result

        result = self.raw_result
        # 位移/长度异常或包数下降时，先尝试在运动预测附近恢复两端实际立面。
        if tracked is None and dt is not None and 0 < dt <= self.reset_gap:
            shift = self._velocity()*dt
            predicted = replace(self.reference, x_front=self.reference.x_front+shift,
                                x_rear=self.reference.x_rear+shift,
                                cabin_length=float(np.median(self.length_history)))
            local = self.detector.recover_local(
                roi_points, predicted, self.position_slack+.5*self.recovery_acceleration*dt*dt,
                self.length_tolerance)
            if local.status == 'ok':
                self._clear_pending()
                return self._accept(local, self._profile(roi_points, local), timestamp, 'local_reacquired')
        if not frame_complete:
            self._clear_pending()
            return replace(result, status='partial_frame', x_front=None, x_rear=None,
                           cabin_length=0., cabin_box=None, track_epoch=self.epoch,
                           track_event='quality_rejected', z_check_status='帧包数下降且两端未恢复')
        if dt is not None and dt > self.reset_gap:
            # 超时后仍保留身份参考，但必须重新确认；预测坐标只用于候选搜索。
            shift = self._velocity() * dt
            predicted = replace(self.reference,
                                x_front=self.reference.x_front + shift,
                                x_rear=self.reference.x_rear + shift)
            recovery = self.detector.detect(
                roi_points, reference=predicted,
                max_position_step=self.position_slack + .5 * self.recovery_acceleration * dt * dt,
                length_tolerance=self.length_tolerance)
            if (recovery.status == 'ok' and self._same_identity(
                    recovery, self._profile(roi_points, recovery), timestamp)):
                result = recovery
        if result.status != "ok":
            self._clear_pending()
            return replace(result, track_epoch=self.epoch, track_event='missing',
                           x_front=None, x_rear=None, cabin_length=0., cabin_box=None)
        profile = self._profile(roi_points, result)
        if profile is None and tracked is None:
            self._clear_pending()
            return replace(result, status='position_rejected', x_front=None, x_rear=None,
                           cabin_length=0., cabin_box=None, track_epoch=self.epoch,
                           track_event='quality_rejected', z_check_status='缺少两端完整立面支持')
        if tracked is not None:
            same_pair = (abs(result.x_front - tracked.x_front) < 1e-6
                         and abs(result.x_rear - tracked.x_rear) < 1e-6)
            if same_pair:
                self._clear_pending()
                return self._accept(tracked, profile, timestamp, 'tracked')

        compatible = False
        if self.pending is not None:
            pending_dt = timestamp - self.pending_time
            limit = self.position_slack + self.max_speed * pending_dt
            compatible = (0 < pending_dt <= self.reset_gap
                          and abs(result.x_front - self.pending.x_front) <= limit
                          and abs(result.x_rear - self.pending.x_rear) <= limit
                          and abs(result.cabin_length - self.pending.cabin_length)
                          <= self.length_tolerance
                          and self._profiles_match(self.pending_profile, profile))
        if not compatible:
            self._candidate_counter += 1
            self.candidate_id = self._candidate_counter
        self.pending_count = self.pending_count + 1 if compatible else 1
        self.pending, self.pending_time = result, timestamp
        self.pending_profile = profile
        self.candidate = result
        if self.pending_count >= self.confirm_frames:
            same_identity = self._same_identity(result, profile, timestamp)
            if not same_identity:
                self.epoch += 1
                self.motion_history.clear()
                self.length_history.clear()
            self.confirmed_candidate_id = self.candidate_id
            self._clear_pending()
            return self._accept(result, profile, timestamp,
                                'resumed' if same_identity else 'reacquired')
        if tracked is not None:
            return self._accept(tracked, self._profile(roi_points, tracked), timestamp, 'tracked')
        return replace(result, status="tracking_acquiring", track_epoch=self.epoch,
                       track_event='confirming', x_front=None, x_rear=None,
                       cabin_length=0., cabin_box=None,
                       z_check_status=f"候选待确认 {self.pending_count}/{self.confirm_frames}")
