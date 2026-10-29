# -*- coding: utf-8 -*-
"""
卡尔曼滤波器模块：
用于 TM16 车辆车厢位置跟踪与高度平滑及缺失帧优化。
提供开关制（enabled）与全部参数可调。

核心设计原则：
1. 统一端点语义：内部严格使用 x_low（小X端）、x_high（大X端），杜绝 front/rear 倒置。
   预测与外推输出永远保证 x_low < x_high。
2. 严格时限与超时：位置与高度外推同时受限于 max_coast_frames 与 max_coast_time，到期立即失效。
3. 外推预测不延长可靠实测时间戳，不反馈为实测。
4. 真实高度阶跃需经多帧相互一致性确认，拒绝盲目接受孤立离群噪点。
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class KalmanConfig:
    enabled: bool = True
    max_coast_frames: int = 5             # 最大预测补帧数（容忍连续缺失帧数）
    max_coast_time: float = 0.60          # 最大预测保持时间 (s)
    pos_process_noise_q: float = 10.0     # 分段恒加速度噪声方差，Q按dt的2/3/4次幂离散
    pos_measure_noise_r: float = 0.15     # 位置测量噪声标准差 R (m)
    height_process_noise_q: float = 0.001 # 高度过程噪声 Q (m^2/s)
    height_measure_noise_r: float = 0.03  # 高度测量噪声标准差 sigma_r (m)，方差 R = sigma_r^2
    height_init_cov: float = 0.10         # 高度初始误差协方差 P0 (m^2)
    height_gate_limit: float = 0.25       # 高度残差门限 (m)：超过此突变拒绝，防止底板噪点
    step_confirm_frames: int = 3          # 真实高度阶跃确认所需连续帧数
    step_consistency_spread: float = 0.15 # 阶跃候选样本内部最大极差 (m)，确保相互一致


class PositionKalmanFilter:
    """
    1D 恒速运动模型卡尔曼滤波：
    状态向量 X = [x_center, v_center, length]^T
    观测向量 Z = [x_center_meas, length_meas]^T

    输出端点严格统一为：
    x_low:  小 X 端 (xc - 0.5 * length)
    x_high: 大 X 端 (xc + 0.5 * length)
    永远满足 x_low < x_high。
    """
    def __init__(self, config: KalmanConfig):
        self.cfg = config
        self.is_initialized = False
        self.last_time: Optional[float] = None
        self.last_meas_time: Optional[float] = None
        self.coast_frames = 0

        # 状态 [x_center, v, length]
        self.x = np.zeros(3, dtype=np.float64)
        # 协方差矩阵 P
        self.P = np.diag([0.2, 1.0, 0.2]).astype(np.float64)

    def reset(self):
        self.is_initialized = False
        self.last_time = None
        self.last_meas_time = None
        self.coast_frames = 0
        self.x[:] = 0.0
        self.P = np.diag([0.2, 1.0, 0.2]).astype(np.float64)

    def current_estimate(self):
        """读取后验位置，不推进时间或消耗预测帧数。"""
        if not self.is_initialized:
            return None
        center, length = float(self.x[0]), abs(float(self.x[2]))
        return center-.5*length, center+.5*length, length

    def predict(self, timestamp: float) -> Tuple[float, float, float]:
        """
        先验预测，返回 (x_low, x_high, length)，保证 x_low < x_high。
        """
        if not self.is_initialized or self.last_time is None:
            raise RuntimeError("PositionKalmanFilter not initialized")

        dt = max(1e-4, timestamp - self.last_time)
        F = np.array([
            [1.0, dt,  0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        q_pos = self.cfg.pos_process_noise_q
        Q = np.array([
            [0.25 * (dt ** 4) * q_pos, 0.5 * (dt ** 3) * q_pos, 0.0],
            [0.5 * (dt ** 3) * q_pos,  (dt ** 2) * q_pos,        0.0],
            [0.0,                     0.0,                      1e-4 * dt]
        ], dtype=np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_time = timestamp

        xc, v, length = self.x[0], self.x[1], abs(self.x[2])
        x_low = xc - 0.5 * length
        x_high = xc + 0.5 * length
        return x_low, x_high, length

    def update(self, x_low: Optional[float] = None, x_high: Optional[float] = None,
               timestamp: float = 0.0, x_front: Optional[float] = None, x_rear: Optional[float] = None):
        """
        实测更新。兼容 (x_low, x_high) 及 (x_front, x_rear)。
        内部强制排序确保 x_low <= x_high。
        """
        xa = x_low if x_low is not None else x_front
        xb = x_high if x_high is not None else x_rear
        if xa is None or xb is None or not (np.isfinite(xa) and np.isfinite(xb)):
            return

        xl = float(min(xa, xb))
        xh = float(max(xa, xb))
        xc_meas = 0.5 * (xl + xh)
        len_meas = xh - xl

        if not self.is_initialized:
            self.x = np.array([xc_meas, 0.0, len_meas], dtype=np.float64)
            self.P = np.diag([0.1, 1.0, 0.1]).astype(np.float64)
            self.last_time = timestamp
            self.last_meas_time = timestamp
            self.coast_frames = 0
            self.is_initialized = True
            return

        if self.last_time is not None and timestamp < self.last_time:
            self.reset()
            self.update(x_low=xl, x_high=xh, timestamp=timestamp)
            return

        dt = max(1e-4, timestamp - (self.last_time or timestamp))
        F = np.array([
            [1.0, dt,  0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        q_pos = self.cfg.pos_process_noise_q
        Q = np.array([
            [0.25 * (dt ** 4) * q_pos, 0.5 * (dt ** 3) * q_pos, 0.0],
            [0.5 * (dt ** 3) * q_pos,  (dt ** 2) * q_pos,        0.0],
            [0.0,                     0.0,                      1e-4 * dt]
        ], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_time = timestamp

        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        r = self.cfg.pos_measure_noise_r
        R = np.diag([r * r, (r * 1.5) ** 2]).astype(np.float64)

        z = np.array([xc_meas, len_meas], dtype=np.float64)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(3, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        self.last_meas_time = timestamp
        self.coast_frames = 0

    def coast(self, timestamp: float) -> Optional[Tuple[float, float, float]]:
        """
        缺失帧预测保持（外推）。
        受限最大外推步数与最大持续时间。若超时返回 None 并重置。
        返回 (x_low, x_high, length)，保证 x_low < x_high。
        注意：外推不更新 last_meas_time。
        """
        if not self.is_initialized or self.last_meas_time is None or self.last_time is None:
            return None

        if timestamp < self.last_time:
            self.reset()
            return None

        self.coast_frames += 1
        elapsed = timestamp - self.last_meas_time
        if self.coast_frames > self.cfg.max_coast_frames or elapsed > self.cfg.max_coast_time:
            self.reset()
            return None

        dt = max(1e-4, timestamp - self.last_time)
        F = np.array([
            [1.0, dt,  0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        q_pos = self.cfg.pos_process_noise_q
        Q = np.array([
            [0.25 * (dt ** 4) * q_pos, 0.5 * (dt ** 3) * q_pos, 0.0],
            [0.5 * (dt ** 3) * q_pos,  (dt ** 2) * q_pos,        0.0],
            [0.0,                     0.0,                      1e-4 * dt]
        ], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_time = timestamp

        xc, v, length = self.x[0], self.x[1], abs(self.x[2])
        x_low = xc - 0.5 * length
        x_high = xc + 0.5 * length
        return x_low, x_high, length


class SingleEndHeightKalman:
    """
    单个车帮端点（小X端或大X端）的高度卡尔曼滤波器。
    状态为标量高度 z。
    带残差门限保护与一致性阶跃确认机制。
    """
    def __init__(self, config: KalmanConfig, z_min: float = 4.0, z_max: float = 7.5):
        self.cfg = config
        self.z_min = z_min
        self.z_max = z_max
        self.step_candidates = deque(maxlen=config.step_confirm_frames + 2)
        self.last_status: str = "uninitialized"
        self.reset()

    def reset(self):
        self.is_initialized = False
        self.z = 0.0
        self.P = self.cfg.height_init_cov
        self.last_time: Optional[float] = None
        self.last_meas_time: Optional[float] = None
        self.coast_frames = 0
        self.last_status = "uninitialized"
        self.step_candidates.clear()

    def predict(self, timestamp: float) -> Optional[float]:
        if not self.is_initialized or self.last_time is None:
            return None
        dt = max(1e-4, timestamp - self.last_time)
        self.P += self.cfg.height_process_noise_q * dt
        self.last_time = timestamp
        return self.z

    def update_detailed(self, z_meas: float, timestamp: float) -> Tuple[Optional[float], str]:
        """
        更新并返回详细状态：(filtered_z, status_string)
        """
        if not np.isfinite(z_meas) or not (self.z_min <= z_meas <= self.z_max):
            coasted = self.coast(timestamp)
            status = "predicted" if coasted is not None else "invalid"
            self.last_status = status
            return coasted, status

        if not self.is_initialized:
            self.z = float(z_meas)
            self.P = self.cfg.height_init_cov
            self.is_initialized = True
            self.last_time = timestamp
            self.last_meas_time = timestamp
            self.coast_frames = 0
            self.step_candidates.clear()
            self.last_status = "reliable_measured"
            return self.z, "reliable_measured"

        if self.last_time is not None and timestamp < self.last_time:
            self.reset()
            return self.update_detailed(z_meas, timestamp)

        self.predict(timestamp)

        # 残差门限（Innovation gating）
        innovation = abs(z_meas - self.z)
        if innovation > self.cfg.height_gate_limit:
            # 记录阶跃候选样本
            self.step_candidates.append((timestamp, float(z_meas)))
            # 清理超期样本（超过0.5s的候选不构成连续阶跃）
            while self.step_candidates and timestamp - self.step_candidates[0][0] > 0.50:
                self.step_candidates.popleft()

            # 检查是否有连续且相互一致的阶跃
            if len(self.step_candidates) >= self.cfg.step_confirm_frames:
                vals = [v for _, v in self.step_candidates]
                spread = max(vals) - min(vals)
                if spread <= self.cfg.step_consistency_spread:
                    # 阶跃确认：连续N帧相互一致
                    self.z = float(np.median(vals))
                    self.P = self.cfg.height_init_cov
                    self.step_candidates.clear()
                    self.last_meas_time = timestamp
                    self.coast_frames = 0
                    self.last_status = "step_confirmed"
                    return self.z, "step_confirmed"

            # 阶跃未确认时，拒绝该离群观测，使用先验预测外推
            coasted = self.coast(timestamp)
            status = "outlier_rejected_coasted" if coasted is not None else "invalid"
            self.last_status = status
            return coasted, status

        # 正常观测，清空阶跃候选
        self.step_candidates.clear()
        r = self.cfg.height_measure_noise_r
        R = r * r
        K = self.P / (self.P + R)
        self.z = self.z + K * (z_meas - self.z)
        self.P = (1.0 - K) * self.P

        self.last_meas_time = timestamp
        self.coast_frames = 0
        self.last_status = "reliable_measured"
        return self.z, "reliable_measured"

    def update(self, z_meas: float, timestamp: float) -> Optional[float]:
        """标量接口，保持向下兼容。"""
        val, _ = self.update_detailed(z_meas, timestamp)
        return val

    def coast(self, timestamp: float) -> Optional[float]:
        """
        外推预测。注意：绝不更新 last_meas_time！
        """
        if not self.is_initialized or self.last_meas_time is None or self.last_time is None:
            return None

        if timestamp < self.last_time:
            self.reset()
            return None

        self.coast_frames += 1
        elapsed = timestamp - self.last_meas_time
        if self.coast_frames > self.cfg.max_coast_frames or elapsed > self.cfg.max_coast_time:
            self.reset()
            return None

        self.predict(timestamp)
        return self.z


class CabinHeightKalmanFilter:
    """
    综合两端高度卡尔曼滤波与一致性裁决：
    独立估计 low (小X端) 与 high (大X端) 的顶部高度。
    返回统一结构：
    (z_low, z_high, diff, z_final, info_low, info_high)
    """
    def __init__(self, config: KalmanConfig, z_min: float = 4.0, z_max: float = 7.5,
                 consistency_threshold: float = 0.30):
        self.cfg = config
        self.consistency_threshold = consistency_threshold
        self.kf_low = SingleEndHeightKalman(config, z_min, z_max)
        self.kf_high = SingleEndHeightKalman(config, z_min, z_max)

    def reset(self):
        self.kf_low.reset()
        self.kf_high.reset()

    def process(self, z_low_raw: Optional[float] = None, z_high_raw: Optional[float] = None,
                timestamp: float = 0.0, z_front_raw: Optional[float] = None,
                z_rear_raw: Optional[float] = None) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        兼容接口：同时支持 (z_low_raw, z_high_raw) 与历史 (z_front_raw, z_rear_raw)。
        按照端点位置严格对应：
        z_low_in  = z_low_raw if z_low_raw is not None else z_front_raw
        z_high_in = z_high_raw if z_high_raw is not None else z_rear_raw
        返回: (z_low, z_high, diff, z_final)
        """
        zl, zh, diff, z_final, _, _ = self.process_detailed(
            z_low_raw=z_low_raw if z_low_raw is not None else z_front_raw,
            z_high_raw=z_high_raw if z_high_raw is not None else z_rear_raw,
            timestamp=timestamp
        )
        return zl, zh, diff, z_final

    def process_detailed(self, z_low_raw: Optional[float], z_high_raw: Optional[float],
                         timestamp: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Dict[str, Any], Dict[str, Any]]:
        """
        完整详细输出，包含两端来源标记与距可靠实测的时间。
        """
        if not self.cfg.enabled:
            return None, None, None, None, {"source": "disabled"}, {"source": "disabled"}

        # 更新或外推 low 端
        info_low = {}
        if z_low_raw is not None and math.isfinite(z_low_raw):
            zl, status_l = self.kf_low.update_detailed(z_low_raw, timestamp)
            info_low["source"] = status_l
            info_low["reason"] = "实测更新" if status_l in ("reliable_measured", "step_confirmed") else "门控拒绝外推"
        else:
            zl = self.kf_low.coast(timestamp)
            info_low["source"] = "predicted" if zl is not None else "invalid"
            info_low["reason"] = "缺测外推" if zl is not None else "外推超时或未初始化"

        info_low["time_since_reliable"] = (
            timestamp - self.kf_low.last_meas_time if self.kf_low.last_meas_time is not None else None
        )
        info_low["coast_frames"] = self.kf_low.coast_frames

        # 更新或外推 high 端
        info_high = {}
        if z_high_raw is not None and math.isfinite(z_high_raw):
            zh, status_h = self.kf_high.update_detailed(z_high_raw, timestamp)
            info_high["source"] = status_h
            info_high["reason"] = "实测更新" if status_h in ("reliable_measured", "step_confirmed") else "门控拒绝外推"
        else:
            zh = self.kf_high.coast(timestamp)
            info_high["source"] = "predicted" if zh is not None else "invalid"
            info_high["reason"] = "缺测外推" if zh is not None else "外推超时或未初始化"

        info_high["time_since_reliable"] = (
            timestamp - self.kf_high.last_meas_time if self.kf_high.last_meas_time is not None else None
        )
        info_high["coast_frames"] = self.kf_high.coast_frames

        diff = None
        z_final = None
        if zl is not None and zh is not None:
            diff = abs(zl - zh)
            if diff <= self.consistency_threshold:
                z_final = 0.5 * (zl + zh)

        return zl, zh, diff, z_final, info_low, info_high
