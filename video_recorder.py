#!/usr/bin/env python3
"""视频片段录制模块 - 检测到人进入禁区时自动录制短视频"""
import cv2
import os
import time
import threading
from datetime import datetime


class VideoRecorder:
    def __init__(self, save_dir, fps=25, duration=10):
        """
        Args:
            save_dir: 视频保存目录
            fps: 录制帧率
            duration: 录制时长（秒）
        """
        self.save_dir = save_dir
        self.fps = fps
        self.duration = duration
        self._recording = False
        self._writer = None
        self._start_time = 0
        self._filepath = ""
        self._lock = threading.Lock()

    @property
    def is_recording(self):
        return self._recording

    @property
    def filepath(self):
        return self._filepath

    def start(self, frame, reason="detection"):
        """开始录制（如果没在录的话）"""
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._start_time = time.time()
            h, w = frame.shape[:2]
            ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            self._filepath = os.path.join(self.save_dir, f"clip_{ts}_{reason}.mp4")
            os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(self._filepath, fourcc, self.fps, (w, h))
            if self._writer.isOpened():
                self._writer.write(frame)
                print(f"[录制] 开始: {self._filepath}")
            else:
                print(f"[录制] 无法创建视频文件: {self._filepath}")
                self._writer = None
                self._recording = False

    def write_frame(self, frame):
        """写入一帧，录制满 duration 秒后自动停止"""
        with self._lock:
            if not self._recording or self._writer is None:
                return
            self._writer.write(frame)
            if time.time() - self._start_time >= self.duration:
                self._writer.release()
                self._writer = None
                self._recording = False
                print(f"[录制] 结束，时长 {self.duration}秒")

    def stop(self):
        """手动停止录制"""
        with self._lock:
            if self._recording and self._writer is not None:
                self._writer.release()
                self._writer = None
                self._recording = False
                print("[录制] 手动停止")
