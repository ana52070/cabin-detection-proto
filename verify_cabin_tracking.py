# -*- coding: utf-8 -*-
"""完整 PCAP 对比：从备份加载旧算法，输出新旧逐帧证据和曲线。"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import types
import zipfile

import numpy as np

import config
from cabin_detector import CabinDetector
from cabin_timeseries import CabinTimeSeriesAnalyzer
from cabin_tracking import CabinPositionTracker
from cabin_measurement import CabinMeasurementCollector
from short_edge_detector import ShortEdgeDetector
from tm16_decoder import TM16Decoder


def old_analyzer(archive):
    with zipfile.ZipFile(archive) as z:
        source = z.read('cabin_timeseries.py').decode('utf-8-sig').replace('\r\n', '\n')
    # 只提取纯数据结构及分析器，避免运行旧窗口和线程。
    start = source.index('@dataclass')
    end = source.index('\n\nclass TimeSeriesComputeWorker', start)
    module = types.ModuleType('baseline_cabin_timeseries')
    import sys
    sys.modules[module.__name__] = module
    exec('import numpy as np\nfrom dataclasses import dataclass\nfrom collections import deque\n'
         'from typing import List, Dict, Optional, Tuple\n' + source[start:end], module.__dict__)
    return module.CabinTimeSeriesAnalyzer()


def extract(index, frame, t0, roi, result):
    row = dict(frame_idx=index, t_rel=frame.timestamp - t0,
               status=result.status, x_front=result.x_front,
               x_rear=result.x_rear, cabin_length=result.cabin_length,
               track_epoch=result.track_epoch)
    for side, x in [('front', result.x_front), ('rear', result.x_rear)]:
        b = roi[np.abs(roi[:, 0] - x) <= .2] if result.status == 'ok' else []
        row[f'z_{side}_raw'] = float(np.percentile(b[:, 2], 98)) if len(b) else None
    return row


def metrics(result, rows):
    points = result['points']
    metrics = {k: v for k, v in result.items() if k != 'points'}
    metrics['position_status'] = dict(Counter(row['status'] for row in rows))
    metrics['final_output_rate'] = result['final_count'] / len(rows)
    # 时间和帧号都连续时才统计跳变，绝不跨缺测连接。
    for attr in ('z_front_med', 'z_rear_med', 'z_final'):
        steps = [abs(getattr(b, attr) - getattr(a, attr))
                 for a, b in zip(points, points[1:])
                 if getattr(a, attr) is not None and getattr(b, attr) is not None
                 and b.frame_idx == a.frame_idx + 1 and 0 < b.t_rel - a.t_rel <= .3]
        metrics[attr] = dict(steps=len(steps), jumps_over_015=sum(x > .15 for x in steps),
                             max_step=max(steps, default=None),
                             p95_step=float(np.percentile(steps, 95)) if steps else None)
    # 恢复等待包含缺测后的预热，非真实高度误差或真值准确率。
    waits = []
    start = None
    measured = False
    for row, p in zip(rows, points):
        valid_pair = (row['status'] == 'ok' and all(
            row[f'z_{s}_raw'] is not None and 4 <= row[f'z_{s}_raw'] <= 7.5
            for s in ('front', 'rear')))
        if not valid_pair:
            start = None
            measured = False
        elif start is None:
            start = p.t_rel
        if p.z_final is not None and start is not None and not measured:
            waits.append(p.t_rel - start)
            measured = True
    metrics['raw_valid_recovery'] = dict(
        segments_with_output=len(waits), max_seconds=max(waits, default=None),
        median_seconds=float(np.median(waits)) if waits else None,
        note='两端原始高度有效段至首次融合输出；未恢复的段不纳入延迟统计，无人工高度真值')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcap', default=config.DEFAULT_TRUCK_PCAP)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('verification_tracking'))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    frames = TM16Decoder().decode_pcap(
        args.pcap, progress_callback=lambda frac, msg: print(msg, flush=True)
        if int(frac * 100) % 20 == 0 else None)
    if not frames:
        raise RuntimeError('PCAP 中没有完整帧')
    print(f'frames={len(frames)}', flush=True)
    roi_detector = ShortEdgeDetector()
    detector = CabinDetector(**config.CABIN_DETECTOR_CONFIG)
    tracker = CabinPositionTracker(CabinDetector(**config.CABIN_DETECTOR_CONFIG),
                                  **config.CABIN_TRACKING_CONFIG)
    collector = CabinMeasurementCollector(tracker.confirm_frames)
    before, after = [], []
    for i, frame in enumerate(frames):
        roi = roi_detector.filter_roi(frame.points_site)
        before.append(extract(i, frame, frames[0].timestamp, roi, detector.detect(roi)))
        tracked = tracker.update(roi, frame.timestamp)
        after.append(collector.collect(i, frame.timestamp - frames[0].timestamp, roi, tracked, tracker))
        if i % 200 == 0:
            print(f'detect={i}/{len(frames)}', flush=True)
    old = old_analyzer(args.baseline).analyze_frames(before)
    new = CabinTimeSeriesAnalyzer(**config.HEIGHT_TIMESERIES_CONFIG).analyze_frames(after)
    report = dict(frames=len(frames), pcap=str(Path(args.pcap).resolve()),
                  pcap_sha256=hashlib.sha256(Path(args.pcap).read_bytes()).hexdigest(),
                  tracking=config.CABIN_TRACKING_CONFIG, filtering=config.HEIGHT_TIMESERIES_CONFIG,
                  before=metrics(old, before), after=metrics(new, after))
    for name, rows, result in [('before', before, old), ('after', after, new)]:
        merged = [dict(row, **asdict(p)) for row, p in zip(rows, result['points'])]
        with (out / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(merged[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                              for k, v in row.items()} for row in merged)
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
    t = [p.t_rel for p in new['points']]
    for ax, attr, title in zip(axes,
                              ['z_front_med', 'z_rear_med', 'diff_med', 'z_final'],
                              ['小X端顶部', '大X端顶部', '两端高度绝对差', '融合高度']):
        for result, label, color in [(old, '修改前', '#aaa'), (new, '修改后', '#168855')]:
            ax.plot(t, [getattr(p, attr) if getattr(p, attr) is not None else np.nan
                        for p in result['points']], '.-', markersize=2, linewidth=.8,
                    label=label, color=color)
        ax.set_ylabel(title + ' (m)')
        ax.grid(alpha=.25)
        ax.legend(loc='lower right')
    axes[2].axhline(.3, color='red', linestyle='--')
    axes[-1].set_xlabel('相对时间 (s)；缺测处断线，不插值')
    fig.suptitle('同一空车PCAP：端点跟踪与因果中值优化前后对比')
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=160)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
