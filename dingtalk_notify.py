#!/usr/bin/env python3
"""
钉钉机器人推送模块
- 使用钉钉自定义机器人 Webhook
- 支持文本、Markdown、图片消息
- 检测到人时自动推送告警

使用方法：
1. 钉钉群 → 设置 → 智能群助手 → 添加机器人 → 自定义机器人
2. 获取 Webhook URL
3. 在 config.ini 中配置 dingtalk_webhook
"""

import json
import time
import threading
import requests
import cv2
import os
import base64
import logging

logger = logging.getLogger('dingtalk')

# Config
WEBHOOK_URL = ""
KEYWORD = "监控"  # 钉钉机器人关键词，消息必须包含此词
ENABLED = False
COOLDOWN = 30  # 推送冷却时间（秒）

_last_push_time = 0
_lock = threading.Lock()


def init(webhook_url, enabled=True, cooldown=30, keyword="监控"):
    """Initialize DingTalk notifier."""
    global WEBHOOK_URL, ENABLED, COOLDOWN, KEYWORD
    WEBHOOK_URL = webhook_url
    ENABLED = enabled
    COOLDOWN = cooldown
    KEYWORD = keyword
    if enabled and webhook_url:
        logger.info(f"[钉钉] 已启用，冷却时间: {cooldown}秒")
    elif not enabled:
        logger.info("[钉钉] 未启用")


def _send_markdown(title, text):
    """Send markdown message to DingTalk group."""
    if not ENABLED or not WEBHOOK_URL:
        return False

    data = {
        "msgtype": "markdown",
        "markdown": {
            "title": title,
            "text": text
        }
    }

    try:
        resp = requests.post(
            WEBHOOK_URL,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        result = resp.json()
        if result.get("errcode") == 0:
            logger.info("[钉钉] 推送成功")
            return True
        else:
            logger.error(f"[钉钉] 推送失败: {result}")
            return False
    except Exception as e:
        logger.error(f"[钉钉] 推送异常: {e}")
        return False


def send_alert(confidence, count, image_path=None):
    """
    Send detection alert to DingTalk.
    Runs in a separate thread to avoid blocking.
    """
    global _last_push_time

    if not ENABLED or not WEBHOOK_URL:
        return

    with _lock:
        now = time.time()
        if now - _last_push_time < COOLDOWN:
            return
        _last_push_time = now

    def _send():
        from datetime import datetime
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        title = f"【{KEYWORD}告警】检测到人员"
        text = f"""
### 🚨 人员检测告警

- **时间**: {now_str}
- **置信度**: {confidence:.1%}
- **累计检测**: {count} 次
- **来源**: USB 摄像头

> 由 DVR-YOLO 自动推送
"""
        _send_markdown(title, text)

    threading.Thread(target=_send, daemon=True).start()


def send_zone_alert(zone_name, count, person_count):
    """Send zone intrusion alert."""
    global _last_push_time

    if not ENABLED or not WEBHOOK_URL:
        return

    with _lock:
        now = time.time()
        if now - _last_push_time < COOLDOWN:
            return
        _last_push_time = now

    def _send():
        from datetime import datetime
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        title = f"【{KEYWORD}告警】区域入侵"
        text = f"""
### ⚠️ 区域入侵告警

- **时间**: {now_str}
- **区域**: {zone_name}
- **当前人数**: {person_count} 人
- **累计进入**: {count} 次

> 由 DVR-YOLO 自动推送
"""
        _send_markdown(title, text)

    threading.Thread(target=_send, daemon=True).start()


def send_daily_summary(total_persons, total_detections, peak_hour):
    """Send daily summary report."""
    if not ENABLED or not WEBHOOK_URL:
        return

    from datetime import datetime
    now_str = datetime.now().strftime('%Y-%m-%d')

    title = f"【{KEYWORD}日报】{now_str}"
    text = f"""
### 📊 每日监控报告 ({now_str})

- **累计检测人数**: {total_persons}
- **累计检测次数**: {total_detections}
- **高峰时段**: {peak_hour}

> 由 DVR-YOLO 自动生成
"""
    _send_markdown(title, text)
