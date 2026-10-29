# -*- coding: utf-8 -*-
"""
PLC Output Adapter for Lingang Site TM16 Measurement
Faithfully mirrors C++ plc_measurements.hpp & plc_registers.hpp:

Convention:
- Truck travel direction is Site +X.
- Large X end (high_x) is Front Board (Head / 车头).
- Small X end (low_x) is Rear Board (Tail / 车尾).
- Formula:
  * plc_head_position_mm = round(high_x * 1000 + lidar_x_mm)  [Reg 43 / VW2586]
  * plc_tail_position_mm = round(low_x * 1000 + lidar_x_mm)   [Reg 44 / VW2588]
  * plc_front_height_mm  = round(front_top_z * 1000)           [Reg 1 / VW2502]
  * plc_rear_height_mm   = round(rear_top_z * 1000)            [Reg 7 / VW2514]
"""

import math
from typing import Optional
from dataclasses import dataclass
import config

@dataclass
class PlcMeasurements:
    head_position_mm: Optional[int] = None   # 大 X 端车头位置 (lidar_x_mm + high_x * 1000)
    tail_position_mm: Optional[int] = None   # 小 X 端车尾位置 (lidar_x_mm + low_x * 1000)
    front_height_mm: Optional[int] = None    # 大 X 端前车帮高度 (front_top_z * 1000)
    rear_height_mm: Optional[int] = None     # 小 X 端后车帮高度 (rear_top_z * 1000)
    valid_mask: int = 0

    @property
    def is_head_valid(self) -> bool:
        return self.head_position_mm is not None

    @property
    def is_tail_valid(self) -> bool:
        return self.tail_position_mm is not None

    @property
    def is_front_h_valid(self) -> bool:
        return self.front_height_mm is not None

    @property
    def is_rear_h_valid(self) -> bool:
        return self.rear_height_mm is not None


def compute_plc_measurements(x_front: Optional[float],
                             x_rear: Optional[float],
                             z_front_raw: Optional[float],
                             z_rear_raw: Optional[float],
                             status: str,
                             lidar_x_mm: float = None) -> PlcMeasurements:
    """
    Computes PLC 4-field measurements from detection results.
    Convention:
      - x_front: 前车帮 (大X端, high_x)
      - x_rear: 后车帮 (小X端, low_x)
      - z_front_raw: 前车帮顶部实测高
      - z_rear_raw: 后车帮顶部实测高
    """
    if lidar_x_mm is None:
        lidar_x_mm = float(config.PLC_CONFIG.get("lidar_x_mm", 5750.0))

    out = PlcMeasurements()
    if status != "ok" or x_front is None or x_rear is None:
        return out

    low_x = min(x_front, x_rear)
    high_x = max(x_front, x_rear)

    def to_mm(val_m: Optional[float], offset_mm: float) -> Optional[int]:
        if val_m is None or not math.isfinite(val_m):
            return None
        rounded = round(val_m * 1000.0 + offset_mm)
        if -2147483648 <= rounded <= 2147483647:
            return int(rounded)
        return None

    # 位置计算：包含 PLC 坐标零点偏移 (默认 5750 mm)
    out.head_position_mm = to_mm(high_x, lidar_x_mm)
    out.tail_position_mm = to_mm(low_x, lidar_x_mm)

    # 高度计算：大X端前车帮高度与小X端后车帮高度（毫米）
    if z_front_raw is not None and math.isfinite(z_front_raw):
        out.front_height_mm = to_mm(z_front_raw, 0.0)
    elif z_rear_raw is not None and math.isfinite(z_rear_raw):
        pass

    if z_rear_raw is not None and math.isfinite(z_rear_raw):
        out.rear_height_mm = to_mm(z_rear_raw, 0.0)

    # 有效位掩码: bit0=front_h, bit1=rear_h, bit2=head_pos, bit3=tail_pos
    mask = 0
    if out.front_height_mm is not None:
        mask |= (1 << 0)
    if out.rear_height_mm is not None:
        mask |= (1 << 1)
    if out.head_position_mm is not None:
        mask |= (1 << 2)
    if out.tail_position_mm is not None:
        mask |= (1 << 3)
    out.valid_mask = mask

    return out
