# -*- coding: utf-8 -*-
"""位置缺点修复的完整录包验证；保留原始边界、校正、恢复、预测及失效来源。"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtTest import QTest
import config
from tm16_decoder import TM16Decoder
from short_edge_detector import ShortEdgeDetector
from cabin_detector import CabinDetector
from cabin_timeseries import CabinTimeSeriesAnalyzer, CabinTimeSeriesWindow, run_cabin_timeseries_pipeline
from verify_focus_timeseries import metrics as height_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'verification_position_dropout'
OUT.mkdir(exist_ok=True)


def position_metrics(rows, start=1726, end=2293):
    rows = [r for r in rows if start <= r['frame_idx'] <= end and not r['is_gap_fill']]
    sources = [r.get('position_type', 'invalid') for r in rows]
    out = {'frames': len(rows), 'sources': {s: sources.count(s) for s in
            ('reliable_measured', 'local_reacquired', 'predicted', 'invalid')}}
    for key in ('x_low', 'x_high', 'cabin_length'):
        valid = [r for r in rows if r.get(key) is not None and r.get('position_type') != 'invalid']
        steps = [abs(b[key]-a[key]) for a, b in zip(rows, rows[1:])
                 if all(r.get(key) is not None and r.get('position_type') != 'invalid' for r in (a,b))
                 and b['frame_idx'] == a['frame_idx']+1 and b['t_rel']-a['t_rel'] <= .3]
        out[key] = dict(valid=len(valid), step_p95_m=float(np.percentile(steps, 95)) if steps else None,
                        step_max_m=max(steps) if steps else None,
                        jumps_over_1m=int(sum(s > 1. for s in steps)),
                        range_m=[min(r[key] for r in valid), max(r[key] for r in valid)] if valid else None)
    intervals = []
    for r in rows:
        source = r.get('position_type', 'invalid')
        if source not in ('predicted', 'invalid', 'local_reacquired'):
            continue
        frame = r['frame_idx']+1
        if intervals and intervals[-1]['source'] == source and intervals[-1]['end_frame'] == frame-1:
            intervals[-1]['end_frame'] = frame
        else:
            intervals.append(dict(source=source, start_frame=frame, end_frame=frame))
    out['intervals'] = intervals
    return out


def main():
    import sys
    render_only = '--render-only' in sys.argv
    cached = '--cached' in sys.argv or render_only
    print('读取保存结果' if cached else '解码完整PCAP', flush=True)
    frames = None if cached else TM16Decoder().decode_pcap(config.DEFAULT_TRUCK_PCAP)
    report = {'pcap': config.DEFAULT_TRUCK_PCAP, 'frame_numbering': '界面从1开始，frame_idx从0开始',
              'cached_analysis': cached, 'total_frames': 2465 if cached else len(frames),
              'kalman_config': dict(config.KALMAN_CONFIG),
              'limits': '最多5帧且不超过0.6秒；关闭卡尔曼时只拒绝异常、不预测',
              'limitation': '无车辆位置和车长人工真值；当前几何坐标仍基于粗网格与局部立面'}
    if not cached:
        selected = [1770,1771,1772,1860,1861,1862,2158,2159,2160,2161,2282,2283,2284]
        np.savez_compressed(OUT/'dropout_clouds.npz', **{f'f{i}': frames[i].points_site for i in selected})
    results = {}
    for enabled in (False, True):
        name = 'on' if enabled else 'off'
        cfg = dict(config.KALMAN_CONFIG, enabled=enabled)
        print('计算', name, flush=True)
        if cached:
            rows = json.loads((OUT/f'after_{name}.json').read_text(encoding='utf-8'))
            result = CabinTimeSeriesAnalyzer(**config.HEIGHT_TIMESERIES_CONFIG, kalman_config=cfg).analyze_frames(rows)
            result['frames_data'] = rows
        else:
            result = run_cabin_timeseries_pipeline(frames, ShortEdgeDetector(),
                      CabinDetector(**config.CABIN_DETECTOR_CONFIG), kalman_config=cfg)
            (OUT/f'after_{name}.json').write_text(json.dumps(result['frames_data'], ensure_ascii=False), encoding='utf-8')
            positions = [dict(frame_idx=i, status=r.status, x_low=r.x_rear, x_high=r.x_front,
                              cabin_length=r.cabin_length, track_event=r.track_event) for i,r in enumerate(result['positions'])]
            (OUT/f'positions_{name}.json').write_text(json.dumps(positions, ensure_ascii=False), encoding='utf-8')
        results[name] = result
        with (OUT/f'after_{name}.csv').open('w', encoding='utf-8-sig', newline='') as fp:
            data = [asdict(p) for p in result['points']]
            writer = csv.DictWriter(fp, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        before = json.loads((ROOT/'verification_focus_1727_2294'/f'after_{name}.json').read_text(encoding='utf-8'))
        baseline = CabinTimeSeriesAnalyzer(**config.HEIGHT_TIMESERIES_CONFIG, kalman_config=cfg).analyze_frames(before)
        report[name] = dict(before_focus=position_metrics(before), after_focus=position_metrics(result['frames_data']),
                            before_full=position_metrics(before, 0, report['total_frames']-1),
                            after_full=position_metrics(result['frames_data'], 0, report['total_frames']-1),
                            before_height=height_metrics(baseline['points'],1726,2293),
                            after_height=height_metrics(result['points'],1726,2293),
                            before_full_height=height_metrics(baseline['points'],0,report['total_frames']-1),
                            after_full_height=height_metrics(result['points'],0,report['total_frames']-1))
        events = [r for r in result['frames_data'] if not r['is_gap_fill'] and r['frame_idx']+1 in (1772,1862,2160,2161,2284)]
        report[name]['events'] = [{k:r.get(k) for k in ('frame_idx','packet_count','packet_reference','frame_complete',
            'x_low_raw','x_high_raw','cabin_length_raw','x_low','x_high','cabin_length','position_type','track_event','position_reason')} for r in events]
        print(name, report[name]['after_focus'], flush=True)
    report['source_sha256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('cabin_detector.py','cabin_tracking.py','cabin_timeseries.py','cabin_kalman.py','visualizer_gui.py','config.py')}
    if not render_only:
        (OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,
                                                 default=lambda v: v.item()),encoding='utf-8')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig, axes = plt.subplots(3,1,figsize=(15,10),sharex=True)
    rows = [r for r in results['on']['frames_data'] if not r['is_gap_fill'] and 1726<=r['frame_idx']<=2370]
    for ax, key, title in zip(axes, ('x_high','x_low','cabin_length'),('前车帮位置','后车帮位置','车长跨度')):
        x = [r['frame_idx']+1 for r in rows]
        raw_key = {'x_high':'x_high_raw','x_low':'x_low_raw','cabin_length':'cabin_length_raw'}[key]
        ax.plot(x,[r.get(raw_key) if r.get(raw_key) is not None else np.nan for r in rows],
                color='#999999',linewidth=.8,label='原始粗聚类')
        ax.plot(x,[r.get(key) if r['position_type'] in ('reliable_measured','local_reacquired') else np.nan for r in rows],
                color='#2563eb',linewidth=1.2,label='接受的实测校正/局部恢复')
        ax.plot(x,[r.get(key) if r['position_type']=='predicted' else np.nan for r in rows],
                color='#f97316',linestyle='None',marker='s',markersize=3,label='限时预测')
        ax.set_ylabel(title+' (m)')
        ax.grid(alpha=.25)
        ax.legend(loc='best')
    axes[-1].set_xlabel('界面帧号（从1开始）')
    fig.suptitle('位置缺点修复：原始结果、接受的测量与限时预测分开显示')
    fig.tight_layout()
    fig.savefig(OUT/'position_comparison.png',dpi=150)
    plt.close(fig)
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont(str(Path(os.environ.get('WINDIR',r'C:\Windows'))/'Fonts'/'msyh.ttc'))
    app.setFont(QFont('Microsoft YaHei',9))
    win = CabinTimeSeriesWindow()
    win._on_worker_finished(results['on'])
    win.tabs.setCurrentIndex(1)
    win.show()
    actual = [p for p in results['on']['points'] if not p.is_gap_fill]
    last_visible = max(p.t_rel for p in actual if p.position_type != 'invalid')
    win.p1.setXRange(actual[1726].t_rel,last_visible+1.5,padding=.01)
    win.set_current_frame(1771)
    for _ in range(4): app.processEvents()
    QTest.qWait(200)
    # 等待隐藏的高度页签与可见位置页签完成时间轴链接和坐标轴布局。
    for plot in win.position_plots:
        plot.setXRange(actual[1726].t_rel,last_visible+1.5,padding=.01)
    QTest.qWait(100)
    win.grab().save(str(OUT/'gui_position.png'))
    win.close()
    print('验证完成，结果位于', OUT, flush=True)


if __name__ == '__main__':
    main()
