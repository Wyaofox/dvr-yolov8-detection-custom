#!/usr/bin/env python3
"""定时报告模块 - 统计检测数据，生成每日汇总，推送到钉钉"""
import time
import threading
import json
import os
from datetime import datetime, timedelta

# Global stats
stats = {
    'total_detections': 0,
    'zone_events': {},      # zone_name -> count
    'line_crossings': {},   # line_name -> {'in': 0, 'out': 0}
    'hourly_counts': [0] * 24,
    'date': '',
}
_lock = threading.Lock()
_report_callback = None
_started = False


def init(callback=None):
    """初始化报告模块，callback(text) 用于推送"""
    global _report_callback, _started
    _report_callback = callback
    stats['date'] = datetime.now().strftime('%Y-%m-%d')
    if not _started:
        _started = True
        t = threading.Thread(target=_report_loop, daemon=True)
        t.start()
        print("[报告] 定时报告已启动，每天 23:00 推送")


def record_detection(hour=None):
    """记录一次检测事件"""
    with _lock:
        stats['total_detections'] += 1
        h = hour if hour is not None else datetime.now().hour
        if 0 <= h < 24:
            stats['hourly_counts'][h] += 1


def record_zone_event(zone_name):
    with _lock:
        stats['zone_events'][zone_name] = stats['zone_events'].get(zone_name, 0) + 1


def record_line_crossing(line_name, direction):
    """direction: 'in' or 'out'"""
    with _lock:
        if line_name not in stats['line_crossings']:
            stats['line_crossings'][line_name] = {'in': 0, 'out': 0}
        stats['line_crossings'][line_name][direction] += 1


def get_stats():
    with _lock:
        return json.loads(json.dumps(stats))


def _report_loop():
    """后台循环，每天 23:00 推送日报"""
    global stats
    while True:
        now = datetime.now()
        target = now.replace(hour=23, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        time.sleep(wait_seconds)

        # Generate and push report
        with _lock:
            report = _generate_report()
            # Reset stats
            stats = {
                'total_detections': 0,
                'zone_events': {},
                'line_crossings': {},
                'hourly_counts': [0] * 24,
                'date': datetime.now().strftime('%Y-%m-%d'),
            }

        if _report_callback:
            try:
                _report_callback(report)
            except Exception as e:
                print(f"[报告] 推送失败: {e}")


def _generate_report():
    total = stats['total_detections']
    hourly = stats['hourly_counts']
    peak_hour = hourly.index(max(hourly)) if total > 0 else -1

    lines = [f"### Daily Report ({stats['date']})"]
    lines.append(f"- Total detections: **{total}**")
    if peak_hour >= 0:
        lines.append(f"- Peak hour: **{peak_hour}:00** ({hourly[peak_hour]} times)")

    if stats['zone_events']:
        lines.append("\n**Zone Events:**")
        for name, count in stats['zone_events'].items():
            lines.append(f"- {name}: {count} times")

    if stats['line_crossings']:
        lines.append("\n**Line Crossings:**")
        for name, counts in stats['line_crossings'].items():
            lines.append(f"- {name}: IN {counts['in']} / OUT {counts['out']}")

    return "\n".join(lines)
