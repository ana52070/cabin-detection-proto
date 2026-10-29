import numpy as np
import struct
import math
import os
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from pcap_reader import PcapReader
from pointcloud_backend import PointCloudBackend
import config

@dataclass
class TM16Frame:
    frame_id: int
    timestamp: float
    points_radar: np.ndarray  # (N, 5): [x, y, z, intensity, ring]
    points_site: np.ndarray   # (N, 5): [x_site, y_site, z_site, intensity, ring]
    packet_count: int
    calibration_state: str
    vertical_angles: np.ndarray # (16,)

    @property
    def point_count(self) -> int:
        return len(self.points_radar)

    def get_points(self, use_site: bool = True) -> np.ndarray:
        return self.points_site if use_site else self.points_radar

    def get_roi_mask(self, roi: dict, use_site: bool = True) -> np.ndarray:
        pts = self.get_points(use_site)
        if len(pts) == 0:
            return np.zeros(0, dtype=bool)
        return (
            (pts[:, 0] >= roi['x_min']) & (pts[:, 0] <= roi['x_max']) &
            (pts[:, 1] >= roi['y_min']) & (pts[:, 1] <= roi['y_max']) &
            (pts[:, 2] >= roi['z_min']) & (pts[:, 2] <= roi['z_max'])
        )

# Precompute timing offsets for 32 channels in a block
CHANNEL_TIMING_OFFSETS = np.zeros(32, dtype=np.float32)
for firing in range(config.FIRINGS_PER_BLOCK):
    for ring in range(config.LASER_COUNT):
        ch = firing * config.LASER_COUNT + ring
        CHANNEL_TIMING_OFFSETS[ch] = ring * config.TM16_DSR_TOFFSET_US + firing * config.TM16_FIRING_TOFFSET_US

def crossed_cut_angle(prev_az: int, curr_az: int, cut_100: int) -> bool:
    if cut_100 >= 36000:
        return False
    delta = (curr_az - prev_az + 36000) % 36000
    if delta == 0 or delta > 18000:
        return False
    to_cut = (cut_100 - prev_az + 36000) % 36000
    return 0 < to_cut <= delta

class TM16Decoder:
    def __init__(self,
                 cut_deg: float = config.DEFAULT_CUT_ANGLE_DEG,
                 extrinsic_matrix: np.ndarray = config.EXTRINSIC_MATRIX,
                 use_calibration: bool = True,
                 backend: Optional[str] = None):
        self.cut_100 = int(round(cut_deg * 100.0)) % 36000
        self.extrinsic_matrix = extrinsic_matrix
        self.use_calibration = use_calibration
        self.backend = PointCloudBackend(backend if backend is not None else os.environ.get('TM16_DECODE_BACKEND', 'auto').lower())
        self.vertical_angles = config.DEFAULT_VERTICAL_ANGLES.copy()
        self.calibration_state = "DEFAULT"
        self._update_trig_tables()
        self.reset()

    @property
    def active_backend(self):
        return self.backend.active_backend

    @property
    def fallback_reason(self):
        return self.backend.fallback_reason

    @property
    def backend_status(self):
        return self.backend.status

    def _update_trig_tables(self):
        v_rad = np.radians(self.vertical_angles)
        self.cos_v = np.cos(v_rad).astype(np.float32)
        self.sin_v = np.sin(v_rad).astype(np.float32)

    def reset(self):
        self.synced = False
        self.has_az = False
        self.last_az = 0
        self.last_diff = 0
        self.packet_count = 0
        self.frame_points = []
        self.frame_count = 0

    def parse_difop(self, payload: bytes) -> bool:
        if len(payload) != config.DIFOP_PACKET_SIZE or payload[0:2] != b'\xa5\xff':
            return False
        if not self.use_calibration:
            return True

        angles = struct.unpack('<16f', payload[834:834+64])
        for i, a in enumerate(angles):
            if not math.isfinite(a) or a < -16.5 + 2 * i or a > -13.5 + 2 * i:
                return False

        new_angles = np.zeros(16, dtype=np.float32)
        for i in range(16):
            ring = config.DIFOP_ANGLE_TO_RING[i]
            new_angles[ring] = angles[i]

        self.vertical_angles = new_angles
        self.calibration_state = "DIFOP"
        self._update_trig_tables()
        return True

    def _transform_to_site(self, pts_radar: np.ndarray) -> np.ndarray:
        if len(pts_radar) == 0:
            return pts_radar.copy()
        xyz_homo = np.hstack([pts_radar[:, :3], np.ones((len(pts_radar), 1), dtype=np.float32)])
        xyz_site = (self.extrinsic_matrix @ xyz_homo.T).T[:, :3]
        return np.column_stack([xyz_site, pts_radar[:, 3:]])

    def _build_frame(self, frame_id: int, timestamp: float) -> TM16Frame:
        # 切帧/坏块处理仍按包顺序进行，只把逐块的小数组运算合为整帧运算。
        blocks = self.frame_points
        packed = np.empty((0, 6), dtype=np.float64)
        if blocks:
            raw = np.frombuffer(b''.join(b[0] for b in blocks),
                                dtype=[('dist', '<u2'), ('intensity', 'u1')]).reshape(-1, 32)
            distances = raw['dist'] * config.DISTANCE_RES
            rows, channels = np.nonzero((raw['dist'] > 0) & (distances >= 0.4) & (distances <= 150.0))
            rings = channels % config.LASER_COUNT
            azimuths = np.asarray([b[1] for b in blocks], dtype=np.float32)[rows]
            differences = np.asarray([b[2] for b in blocks], dtype=np.float32)[rows]
            angles = (azimuths + differences * (CHANNEL_TIMING_OFFSETS[channels] / config.TM16_BLOCK_TDURATION_US)) % 36000.0 * 0.01
            # 每块保留当时的标定表引用，避免帧中途 DIFOP 更新影响先前点。
            cos_v = np.stack([b[3] for b in blocks])[rows, rings]
            sin_v = np.stack([b[4] for b in blocks])[rows, rings]
            packed = np.column_stack((distances[rows, channels], angles, cos_v, sin_v,
                                      raw['intensity'][rows, channels], rings))
        all_pts_radar, all_pts_site = self.backend.compute(packed, self.extrinsic_matrix)
        return TM16Frame(
            frame_id=frame_id,
            timestamp=timestamp,
            points_radar=all_pts_radar,
            points_site=all_pts_site,
            packet_count=self.packet_count,
            calibration_state=self.calibration_state,
            vertical_angles=self.vertical_angles.copy()
        )

    def decode_pcap(self,
                    pcap_path: str,
                    max_frames: Optional[int] = None,
                    progress_callback: Optional[Callable[[float, str], None]] = None
                    ) -> List[TM16Frame]:
        reader = PcapReader(pcap_path)
        frames: List[TM16Frame] = []
        self.reset()

        saved_diff = 0
        current_frame_time = 0.0

        def on_pcap_progress(frac: float):
            if progress_callback:
                progress_callback(frac, f"解析 PCAP：{int(frac * 100)}%（{len(frames)} 帧）| {self.backend_status}")

        on_pcap_progress(0.0)

        for ts, src_ip, dst_port, payload in reader.read_packets(progress_callback=on_pcap_progress):
            if dst_port == config.DIFOP_PORT:
                self.parse_difop(payload)
                continue

            if dst_port != config.MSOP_PORT or len(payload) != config.MSOP_PACKET_SIZE:
                continue

            if payload[0:2] != b'\xff\xee':
                self.reset()
                continue

            for block in range(config.BLOCKS_PER_PACKET):
                b_off = block * config.BLOCK_SIZE
                if payload[b_off:b_off+2] != b'\xff\xee':
                    self.reset()
                    continue

                az = struct.unpack('<H', payload[b_off+2:b_off+4])[0]
                if az >= 36000:
                    self.reset()
                    continue

                if self.has_az and crossed_cut_angle(self.last_az, az, self.cut_100):
                    if self.synced and len(self.frame_points) > 0:
                        frame = self._build_frame(len(frames), current_frame_time)
                        if frame.point_count:
                            frames.append(frame)
                            if max_frames and len(frames) >= max_frames:
                                on_pcap_progress(1.0)
                                return frames

                    self.frame_points = []
                    self.packet_count = 0
                    self.synced = True
                    current_frame_time = ts

                diff = 0
                if block < config.BLOCKS_PER_PACKET - 1:
                    next_off = (block + 1) * config.BLOCK_SIZE
                    if payload[next_off:next_off+2] == b'\xff\xee':
                        next_az = struct.unpack('<H', payload[next_off+2:next_off+4])[0]
                        if next_az < 36000:
                            diff = (next_az - az + 36000) % 36000
                            if diff > 18000:
                                diff = 0
                else:
                    diff = saved_diff

                if 0 < diff <= 18000:
                    saved_diff = diff
                    self.last_diff = diff

                if self.synced:
                    self.frame_points.append((payload[b_off+4:b_off+100], az, diff, self.cos_v, self.sin_v))

                self.last_az = az
                self.has_az = True

            if self.synced:
                self.packet_count += 1

        if self.synced and len(self.frame_points) > 0:
            frame = self._build_frame(len(frames), current_frame_time)
            if frame.point_count:
                frames.append(frame)

        on_pcap_progress(1.0)
        return frames
