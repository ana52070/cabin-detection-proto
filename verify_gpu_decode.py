"""对同一 PCAP 验证完整输出，并保存原版/批量 CPU/GPU 的实测时间。"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import config
from tm16_decoder import TM16Decoder
from verification_gpu.baseline_decoder import TM16Decoder as ReferenceDecoder


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def signature(frame):
    return dict(frame_id=frame.frame_id, timestamp=frame.timestamp,
                packet_count=frame.packet_count, point_count=frame.point_count,
                calibration_state=frame.calibration_state,
                vertical_angles=frame.vertical_angles.tolist(),
                attributes=digest(frame.points_radar[:, 3:]),
                radar=digest(frame.points_radar), site=digest(frame.points_site))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcap', default=config.DEFAULT_TRUCK_PCAP)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--out', default='verification_gpu/metrics.json')
    args = parser.parse_args()
    print('解析原版基线……', flush=True)
    start = time.perf_counter()
    reference = ReferenceDecoder().decode_pcap(args.pcap, max_frames=args.max_frames)
    reference_seconds = time.perf_counter() - start
    signatures = [signature(frame) for frame in reference]
    # 仅保留基线坐标及摘要，减少同时持有三套整文件点云的内存占用。
    coordinates = [np.column_stack((f.points_radar[:, :3], f.points_site[:, :3])) for f in reference]
    del reference
    gc.collect()
    report = dict(pcap=str(Path(args.pcap).resolve()), bytes=Path(args.pcap).stat().st_size,
                  max_frames=args.max_frames, frames=len(signatures),
                  points=sum(f['point_count'] for f in signatures),
                  baseline_seconds=reference_seconds, runs=[])
    for mode, label in [('cpu', '批量 CPU'), ('gpu', 'GPU 首次'), ('gpu', 'GPU 再次')]:
        print(f'解析{label}……', flush=True)
        start = time.perf_counter()
        decoder = TM16Decoder(backend=mode)
        initialized = time.perf_counter()
        frames = decoder.decode_pcap(args.pcap, max_frames=args.max_frames)
        completed = time.perf_counter()
        assert len(frames) == len(signatures), '帧数发生变化'
        maximum_error = 0.
        identical = True
        for frame, expected, expected_xyz in zip(frames, signatures, coordinates):
            actual = signature(frame)
            identical &= actual == expected
            for field in ('frame_id', 'timestamp', 'packet_count', 'point_count',
                          'calibration_state', 'vertical_angles', 'attributes'):
                assert actual[field] == expected[field], f'第 {frame.frame_id} 帧 {field} 不一致'
            xyz = np.column_stack((frame.points_radar[:, :3], frame.points_site[:, :3]))
            error = float(np.max(np.abs(xyz - expected_xyz))) if len(xyz) else 0.
            maximum_error = max(maximum_error, error)
        assert maximum_error <= 5e-5, f'坐标最大误差超限：{maximum_error} m'
        if mode == 'cpu':
            assert identical, 'CPU 输出未保持逐值一致'
        if mode == 'gpu':
            assert decoder.active_backend == 'gpu', decoder.fallback_reason
        run = dict(label=label, requested=mode, active=decoder.active_backend,
                   status=decoder.backend_status, fallback_reason=decoder.fallback_reason,
                   init_seconds=initialized - start, decode_seconds=completed - initialized,
                   total_seconds=completed - start, speedup=reference_seconds / (completed - start),
                   byte_identical=identical, max_coordinate_error_m=maximum_error)
        report['runs'].append(run)
        print(json.dumps(run, ensure_ascii=False), flush=True)
        del frames, decoder
        gc.collect()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'验证通过：{output}', flush=True)


if __name__ == '__main__':
    main()
