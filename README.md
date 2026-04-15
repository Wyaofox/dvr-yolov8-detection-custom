# YOLO实时监控检测系统（二次开发版）

> 基于 [dvr-yolov8-detection](https://github.com/FlyingFathead/dvr-yolov8-detection) 进行二次开发

## 项目简介

本项目是一个基于YOLOv8/YOLOv11的实时监控检测系统，通过摄像头实时捕捉画面，自动检测画面中的人员，并对指定区域进行入侵检测和驻留报警。主要用于解决传统监控需要人工盯屏效率低的问题。

## 新增功能（二次开发）

在原项目基础上，我们新增了以下核心功能：

### 1. 区域入侵检测
- 用户可在Web界面上绘制多边形监控区域
- 自动计算人体边界框与区域的重叠度（IoU）
- 重叠度超过阈值（默认30%）判定为入侵

### 2. 驻留报警机制
- 防误报设计：只有人员在区域内停留超过3秒才触发报警
- 路过人员不触发报警，减少误报率
- 支持多人同时在区域内，各自独立计时

### 3. NiceGUI管理界面
- 替代原Flask界面，更现代化的Web管理界面（端口5001）
- **检测记录页**：实时显示检测人员信息、track_id、置信度
- **区域管理页**：点击绘制多边形区域，支持热更新
- **录制查看页**：展示自动保存的报警录像文件
- **系统信息页**：显示FPS、GPU状态、区域统计

### 4. 自动视频录制
- 检测到入侵报警时自动录制10秒视频片段
- 视频保存为MP4格式，文件名带时间戳

### 5. 钉钉推送（预留）
- 支持钉钉机器人Webhook推送告警（接口已预留）
- 可推送区域入侵报警、每日统计日报

## 技术栈

| 技术 | 作用 | GitHub |
|------|------|--------|
| YOLOv11 (ultralytics) | 目标检测模型 | https://github.com/ultralytics/ultralytics |
| supervision | 检测辅助（标注、跟踪、区域） | https://github.com/roboflow/supervision |
| OpenCV | 视频采集与图像处理 | https://github.com/opencv/opencv |
| NiceGUI | Web管理界面 | https://github.com/zauberzeug/nicegui |

## 项目结构

```
dvr-yolov8-detection/
├── yolov8_live_rtmp_stream_detection.py  # 主程序
├── zone_manager.py                        # 区域检测模块
├── video_recorder.py                      # 视频录制模块
├── nicegui_dashboard.py                   # NiceGUI界面
├── dingtalk_notify.py                     # 钉钉推送模块
├── daily_report.py                        # 每日统计模块
├── config.ini                             # 配置文件
├── start.bat                              # Windows一键启动
└── docs/                                  # 答辩文档
```

## 快速启动

### Windows

```powershell
# 双击运行 start.bat 或命令行执行
start.bat
```

启动后会自动打开浏览器访问 `http://127.0.0.1:5001`

### 配置说明

编辑 `config.ini` 文件：

```ini
[zones]
enabled = true
clip_duration = 10
polygons = gate=45,67;41,85;69,86;64,67 | door=46,11;64,11;63,65;47,65
```

区域坐标使用百分比，自动适配分辨率。

## 开发团队

| 角色 | 负责内容 |
|------|---------|
| 组长 | 项目统筹、架构设计、模块整合 |
| 目标检测 | YOLO模型开发与调优 |
| 区域检测 | 入侵检测与驻留报警算法 |
| Web界面 | NiceGUI管理界面开发 |
| 资料整理 | 文献查阅、技术资料整理 |
| PPT制作 | 答辩PPT设计与制作 |
| 讲稿撰写 | 答辩讲稿、演示文稿撰写 |

## 致谢

本项目基于 [FlyingFathead/dvr-yolov8-detection](https://github.com/FlyingFathead/dvr-yolov8-detection) 开源项目进行二次开发，感谢原作者提供的优秀框架。

---

## 原项目说明

`dvr-yolov8-detection` is designed for real-time detection of humans, animals, or objects using the YOLOv8 model and OpenCV. Contrary to its name, we're now supporting models up to **YOLOv11**!

The program supports real-time video streams via RTMP or USB webcams, includes CUDA GPU acceleration for enhanced performance, and provides options for saving detections, triggering alerts and logging events.

### Features

- **Real-time human/animal/object detection and alert system**
- **(New!)** Now runs on the newest YOLOv11 model by default
- Runs on **Python + YOLOv8-11 + OpenCV2**
- Both GUI and headless web server versions (Flask)
- **Supports CUDA GPU acceleration**, CPU-only mode is also supported
- **RTMP streams** or **USB webcams** can be used for real-time video sources
- Set up separate minimum confidence zones with the included masking tool
- Name your regions and get alerted with zone names (i.e. on Telegram)
- Detections can be automatically saved as images with a detection log

### Requirements

- **Python 3.6+** (Python 3.10.x recommended)
- **FFmpeg**
- **CUDA 11.8+** (optional, for GPU acceleration)

### Installation

```bash
git clone https://github.com/Wyaofox/dvr-yolov8-detection-custom.git
cd dvr-yolov8-detection-custom
pip install -r requirements.txt
```

For more details, see the original [dvr-yolov8-detection](https://github.com/FlyingFathead/dvr-yolov8-detection) repository.