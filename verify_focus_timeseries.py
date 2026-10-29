# -*- coding: utf-8 -*-
"""完整录包回放，比较修改前后，重点评估界面第1727～2294帧。"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import csv
import hashlib
import json
import sys
import types
import zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
import config
from tm16_decoder import TM16Decoder
from cabin_detector import CabinDetector
from short_edge_detector import ShortEdgeDetector
from cabin_timeseries import CabinTimeSeriesAnalyzer, CabinTimeSeriesWindow, run_cabin_timeseries_pipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'verification_focus_1727_2294'
OUT.mkdir(exist_ok=True)
FIELDS = ('z_low_filtered', 'z_high_filtered', 'z_final')


def metrics(points, start=0, end=2464):
    points = [p for p in points if start <= p.frame_idx <= end and not p.is_gap_fill]
    result = {'frames': len(points)}
    for key in FIELDS:
        steps = [abs(getattr(b, key)-getattr(a, key))*1000
                 for a, b in zip(points, points[1:])
                 if b.frame_idx == a.frame_idx+1 and b.t_rel-a.t_rel <= .3
                 and getattr(a, key) is not None and getattr(b, key) is not None]
        vals = [getattr(p, key) for p in points if getattr(p, key) is not None]
        result[key] = dict(valid=len(vals), step_p95_mm=float(np.percentile(steps, 95)) if steps else None,
                           step_max_mm=max(steps) if steps else None,
                           jumps_over_150mm=sum(v > 150 for v in steps),
                           range_m=[min(vals), max(vals)] if vals else None)
    for side in ('low', 'high'):
        sources = [getattr(p, 'point_type_'+side) for p in points]
        result['sources_'+side] = {k: sources.count(k) for k in
                                  ('reliable_measured', 'local_reacquired', 'predicted', 'invalid')}
        intervals = []
        for p in points:
            source = getattr(p, 'point_type_'+side)
            if source not in ('predicted', 'invalid'):
                continue
            if intervals and intervals[-1]['source'] == source and intervals[-1]['end_frame'] == p.frame_idx:
                intervals[-1]['end_frame'] = p.frame_idx+1
            else:
                intervals.append(dict(source=source, start_frame=p.frame_idx+1, end_frame=p.frame_idx+1))
        result['intervals_'+side] = intervals
    return result


def main():
    # 从备份加载原分析器，使前后比较包含原始代码行为，而不重写基线。
    with zipfile.ZipFile(ROOT/'backups'/'目标区间优化前_20261021.zip') as archive:
        old = types.ModuleType('focus_baseline')
        sys.modules[old.__name__] = old
        exec(compile(archive.read('cabin_timeseries.py'), 'baseline/cabin_timeseries.py', 'exec'), old.__dict__)
        old_config = types.ModuleType('focus_baseline_config')
        exec(archive.read('config.py'), old_config.__dict__)
    before = {}
    for enabled in (False, True):
        name = 'on' if enabled else 'off'
        rows = json.loads((OUT/f'before_{name}.json').read_text(encoding='utf-8'))
        before[name] = old.CabinTimeSeriesAnalyzer(
            **old_config.HEIGHT_TIMESERIES_CONFIG,
            kalman_config=dict(old_config.KALMAN_CONFIG, enabled=enabled)).analyze_frames(rows)
    cached = '--cached' in sys.argv
    print('读取已保存逐帧结果……' if cached else '解码完整PCAP……', flush=True)
    frames = range(2465) if cached else TM16Decoder().decode_pcap(config.DEFAULT_TRUCK_PCAP)
    print(f'共{len(frames)}帧，开始修改后双模式回放', flush=True)
    after = {}
    report = dict(frame_numbering='界面从1开始；CSV frame_idx从0开始；重点区间含两端共568帧',
                  pcap=config.DEFAULT_TRUCK_PCAP, total_frames=len(frames),
                  height_config=config.BOARD_HEIGHT_CONFIG, kalman_config=config.KALMAN_CONFIG,
                  limitation='没有人工真值；抖动和覆盖率不能等同于绝对测量准确率。位置及车长仍为粗聚类边界和跨度。')
    report['before_off_endpoint_warning'] = '原关闭模式低/高端显示颠倒；before_off指标保留原显示行为。物理端点比较需交换两端。'
    report['cached_analysis'] = cached
    for enabled in (False, True):
        name = 'on' if enabled else 'off'
        result = (CabinTimeSeriesAnalyzer(**config.HEIGHT_TIMESERIES_CONFIG,
                                         kalman_config=dict(config.KALMAN_CONFIG, enabled=enabled)).analyze_frames(
                                             json.loads((OUT/f'after_{name}.json').read_text(encoding='utf-8')))
                  if cached else run_cabin_timeseries_pipeline(
                      frames, ShortEdgeDetector(), CabinDetector(**config.CABIN_DETECTOR_CONFIG),
                      kalman_config=dict(config.KALMAN_CONFIG, enabled=enabled)))
        after[name] = result
        if not cached:
            (OUT/f'after_{name}.json').write_text(json.dumps(result['frames_data'], ensure_ascii=False), encoding='utf-8')
        with (OUT/f'after_{name}.csv').open('w', newline='', encoding='utf-8-sig') as fp:
            rows = [asdict(p) for p in result['points']]
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for stage, data in (('before', before[name]), ('after', result)):
            report[f'{stage}_{name}'] = dict(focus=metrics(data['points'], 1726, 2293),
                                             full=metrics(data['points'], 0, len(frames)-1))
        print(name, report[f'after_{name}']['focus'], flush=True)
    report['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                              ('config.py', 'cabin_timeseries.py', 'cabin_measurement.py', 'visualizer_gui.py')}
    report['common_valid_focus'] = {}
    for key in FIELDS:
        a = {p.frame_idx: p for p in before['on']['points'] if not p.is_gap_fill}
        b = {p.frame_idx: p for p in after['on']['points'] if not p.is_gap_fill}
        pairs = [i for i in range(1727, 2294) if all(
            getattr(series[j], key) is not None for series in (a, b) for j in (i-1, i))
            and all(series[i].t_rel-series[i-1].t_rel <= .3 for series in (a, b))]
        report['common_valid_focus'][key] = dict(adjacent_pairs=len(pairs), **{
            stage+'_p95_mm': float(np.percentile([abs(getattr(series[i], key)-getattr(series[i-1], key))*1000
                                                   for i in pairs], 95)) if pairs else None
            for stage, series in (('before', a), ('after', b))})
    (OUT/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    for label, result, color in (('修改前：卡尔曼开启', before['on'], '#999999'),
                                  ('修改后：中值', after['off'], '#3b82f6'),
                                  ('修改后：卡尔曼', after['on'], '#f97316')):
        points = [p for p in result['points'] if 1726 <= p.frame_idx <= 2293 and not p.is_gap_fill]
        for ax, key in zip(axes, ('z_low_filtered', 'z_high_filtered', 'diff_filtered', 'z_final')):
            # 缺测保留为NaN，不跨无效帧连线。
            ax.plot([p.frame_idx+1 for p in points],
                    [getattr(p, key) if getattr(p, key) is not None else np.nan for p in points],
                    label=label, color=color, linewidth=1.)
    for ax, title in zip(axes, ('小X端顶部 Z', '大X端顶部 Z', '两端绝对差', '综合高度')):
        ax.set_ylabel(title+' (m)')
        ax.grid(alpha=.25)
        ax.legend(loc='best')
    axes[2].axhline(.3, color='red', linestyle='--', linewidth=.8)
    axes[-1].set_xlabel('界面帧号（从1开始）')
    fig.suptitle('第1727～2294帧：统一测高来源前后对比（无实测真值）')
    fig.tight_layout()
    fig.savefig(OUT/'focus_comparison.png', dpi=150)
    plt.close(fig)
    app = QApplication.instance() or QApplication([])
    # Windows的offscreen平台不自动加载系统字体，只影响本脚本的离屏截图。
    font_path = Path(os.environ.get('WINDIR', r'C:\Windows'))/'Fonts'/'msyh.ttc'
    if font_path.exists():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont('Microsoft YaHei', 9))
    win = CabinTimeSeriesWindow()
    win._on_worker_finished(after['on'])
    win.show()
    focus_points = [p for p in after['on']['points'] if 1726 <= p.frame_idx <= 2293 and not p.is_gap_fill]
    win.p1.setXRange(focus_points[0].t_rel, focus_points[-1].t_rel, padding=.01)
    win.set_current_frame(1786)
    for _ in range(5):
        app.processEvents()
    win.grab().save(str(OUT/'gui_height.png'))
    win.tabs.setCurrentIndex(1)
    for _ in range(5):
        app.processEvents()
    win.grab().save(str(OUT/'gui_position.png'))
    win.close()
    print('指标、CSV、对比图和界面截图已保存', flush=True)


if __name__ == '__main__':
    main()
