# -*- coding: utf-8 -*-
"""同一PCAP对比上一版与并行预热/身份恢复版，保留各帧证据。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile
from dataclasses import asdict

import numpy as np

import config
from cabin_detector import CabinDetector
from cabin_tracking import CabinPositionTracker
from cabin_measurement import CabinMeasurementCollector
from cabin_timeseries import CabinTimeSeriesAnalyzer
from short_edge_detector import ShortEdgeDetector
from tm16_decoder import TM16Decoder
from verify_cabin_tracking import old_analyzer, extract, metrics


def baseline_module(archive, filename, name):
    with zipfile.ZipFile(archive) as z:
        source = z.read(filename).decode('utf-8-sig')
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f'{archive}!{filename}', 'exec'), module.__dict__)
    return module


def save_rows(path, rows, result):
    merged = [dict(row, **asdict(point)) for row, point in zip(rows, result['points'])]
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                          for k, v in row.items()} for row in merged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--pcap', default=config.DEFAULT_TRUCK_PCAP)
    parser.add_argument('--output', default=Path('verification_recovery'), type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    old_config = baseline_module(args.baseline, 'config.py', 'baseline_recovery_config')
    old_tracker_cls = baseline_module(args.baseline, 'cabin_tracking.py', 'baseline_recovery_tracking').CabinPositionTracker
    frames = TM16Decoder().decode_pcap(args.pcap, progress_callback=lambda frac, msg:
                                     print(msg, flush=True) if int(frac*100) % 20 == 0 else None)
    if not frames:
        raise RuntimeError('没有完整点云帧')
    roi_filter = ShortEdgeDetector()
    old_tracker = old_tracker_cls(CabinDetector(**old_config.CABIN_DETECTOR_CONFIG),
                                  **old_config.CABIN_TRACKING_CONFIG)
    tracker = CabinPositionTracker(CabinDetector(**config.CABIN_DETECTOR_CONFIG),
                                  **config.CABIN_TRACKING_CONFIG)
    collector = CabinMeasurementCollector(tracker.confirm_frames)
    before, after = [], []
    for i, frame in enumerate(frames):
        roi = roi_filter.filter_roi(frame.points_site)
        previous = old_tracker.update(roi, frame.timestamp)
        before.append(extract(i, frame, frames[0].timestamp, roi, previous))
        current = tracker.update(roi, frame.timestamp)
        after.append(collector.collect(i, frame.timestamp-frames[0].timestamp, roi, current, tracker))
        if i % 200 == 0:
            print(f'detect={i}/{len(frames)}', flush=True)
    (out/'before_frames_data.json').write_text(json.dumps(before, ensure_ascii=False), encoding='utf-8')
    (out/'frames_data.json').write_text(json.dumps(after, ensure_ascii=False), encoding='utf-8')
    previous_analyzer = old_analyzer(args.baseline)
    for key, value in old_config.HEIGHT_TIMESERIES_CONFIG.items():
        setattr(previous_analyzer, key, value)
    old = previous_analyzer.analyze_frames(before)
    new = CabinTimeSeriesAnalyzer(**config.HEIGHT_TIMESERIES_CONFIG).analyze_frames(after)
    focus = {}
    for name, rows, result in [('before', before, old), ('after', after, new)]:
        save_rows(out / f'{name}.csv', rows, result)
        first = next((i for i in range(809, min(len(rows), 840))
                      if result['points'][i].z_final is not None), None)
        focus[name] = dict(first_output_frame=first,
                           wait_seconds=rows[first]['t_rel']-rows[809]['t_rel'] if first is not None else None,
                           frames=[dict(row, **asdict(p)) for row, p in zip(rows[807:817], result['points'][807:817])])
    report = dict(frames=len(frames), baseline=str(args.baseline.resolve()),
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  pcap_sha256=hashlib.sha256(Path(args.pcap).read_bytes()).hexdigest(),
                  tracking=config.CABIN_TRACKING_CONFIG, filtering=config.HEIGHT_TIMESERIES_CONFIG,
                  before=metrics(old, before), after=metrics(new, after), recovery_809=focus)
    (out/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # 保存实际输入供复核确认时的种子，不借用未来检测或重写过去输出。
    (out/'frames_data.json').write_text(json.dumps(after, ensure_ascii=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    for filename, limits in [('comparison.png', None), ('recovery_809.png', (81.5, 86.))]:
        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        for ax, attr, title in zip(axes, ['z_front_med', 'z_rear_med', 'diff_med', 'z_final'],
                                  ['小X端顶部', '大X端顶部', '两端高度差', '融合高度']):
            for result, label, color in [(old, '上一版', '#999'), (new, '恢复优化后', '#168855')]:
                ax.plot([p.t_rel for p in result['points']],
                        [getattr(p, attr) if getattr(p, attr) is not None else np.nan
                         for p in result['points']], '.-', markersize=3, linewidth=.8, label=label, color=color)
            ax.set_ylabel(title+' (m)')
            ax.grid(alpha=.25)
            ax.legend(loc='best')
            if limits:
                ax.set_xlim(*limits)
                ax.axvspan(before[808]['t_rel'], before[809]['t_rel'], alpha=.08, color='red')
                values = [getattr(p, attr) for result in (old, new) for p in result['points']
                          if limits[0] <= p.t_rel <= limits[1] and getattr(p, attr) is not None]
                if values:
                    lo, hi = min(values), max(values)
                    margin = max((hi-lo)*.2, .01)
                    ax.set_ylim(lo-margin, hi+margin)
        axes[-1].set_xlabel('相对时间 (s)；缺测不插值、不回填历史输出')
        fig.suptitle('目标确认与高度预热并行：上一版与本轮对比')
        fig.tight_layout()
        fig.savefig(out/filename, dpi=150)
        plt.close(fig)
    brief = {k: v for k, v in report.items() if k != 'recovery_809'}
    brief['recovery_809'] = {k: {a:b for a,b in v.items() if a != 'frames'} for k,v in focus.items()}
    print(json.dumps(brief, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
