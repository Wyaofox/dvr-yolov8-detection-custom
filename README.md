# YOLO实时监控检测系统

> 基于 [dvr-yolov8-detection](https://github.com/FlyingFathead/dvr-yolov8-detection) 二次开发

一个基于YOLOv8/YOLOv11的实时监控检测系统，支持区域入侵检测、驻留报警、自动录制、Web管理界面。

---

## 功能特点

| 功能 | 说明 |
|------|------|
| **实时人体检测** | 基于YOLOv11模型，实时检测画面中的人员 |
| **区域入侵检测** | 用户在界面绘制监控区域，自动检测人员入侵 |
| **驻留报警** | 人员在区域停留超过3秒才报警，路过不触发 |
| **自动录制** | 检测到入侵时自动录制10秒视频片段 |
| **Web管理界面** | NiceGUI界面（端口5001），支持实时查看、区域管理 |
| **钉钉推送** | 预留钉钉机器人Webhook接口 |

---

## 最终效果

启动后自动打开浏览器，访问管理界面 `http://127.0.0.1:5001`：

### 1. 实时检测画面
```
画面左上角显示：
- FPS: 20.5
- 检测到的人员边界框（绿色）
- track_id 标签（跟踪编号）
- 置信度分数（0.87）
```

### 2. 区域入侵检测
```
画面下方显示：
- #1 gate: 1 in (15 total) max:3.2s
- #2 door: 0 in (8 total)

解释：
- gate区域当前有1人，累计进入15人次，最大驻留3.2秒
- door区域当前无人，累计进入8人次
```

### 3. 驻留报警触发
```
当人员在区域内停留超过3秒：
- 控制台打印: [报警] gate区域入侵，人数:1
- 自动开始录制视频（10秒）
- 画面上方显示红色ALARM标志
```

### 4. Web管理界面
```
Tab 1 - 检测记录：显示每位检测人员的track_id、置信度、进入时间
Tab 2 - 区域管理：在画面上点击绘制多边形区域
Tab 3 - 录制查看：查看保存的报警视频文件列表
Tab 4 - 系统信息：显示FPS、GPU状态、区域统计
```

---

## 安装教程

### 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 或 Linux |
| Python | 3.10+ |
| GPU | NVIDIA显卡（可选，用于加速） |
| 显存 | 4GB+（推荐） |

### 步骤1：克隆项目

```bash
git clone https://github.com/Wyaofox/dvr-yolov8-detection-custom.git
cd dvr-yolov8-detection-custom
```

### 步骤2：安装Python依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `ultralytics` — YOLO模型
- `opencv-python` — 视频处理
- `supervision` — 检测辅助
- `nicegui` — Web界面
- `numpy` — 数值计算

### 步骤3：下载YOLO模型

首次运行会自动下载模型，或手动下载：

```bash
# 自动下载（首次运行时）
python yolov8_live_rtmp_stream_detection.py

# 或手动下载到项目目录
# 下载地址: https://github.com/ultralytics/assets/releases
# 文件: yolo11m.pt（约50MB）
```

### 步骤4：配置摄像头

编辑 `config.ini`：

```ini
[input]
use_webcam = true
webcam_index = 0    # 摄像头编号，0为默认摄像头
```

### 步骤5：启动系统

**Windows：**
```bash
# 双击运行
start.bat

# 或命令行
python yolov8_live_rtmp_stream_detection.py
```

**Linux：**
```bash
python yolov8_live_rtmp_stream_detection.py
```

启动后会自动打开浏览器访问 `http://127.0.0.1:5001`

---

## 使用教程

### 1. 查看实时检测

启动后，OpenCV窗口和Web界面都会显示实时检测画面：
- 绿色框：检测到的人员边界框
- 数字标签：track_id（跟踪编号）
- 置信度：0.XX格式

### 2. 绘制监控区域

在Web界面中：
1. 点击 **区域管理** Tab
2. 在画面上点击鼠标添加顶点
3. 至少点击3个点形成多边形
4. 输入区域名称（如 `gate`）
5. 点击 **保存区域**

区域会立即生效，无需重启系统。

### 3. 测试入侵检测

1. 让人员进入绘制的区域
2. 观察画面下方显示的区域统计
3. 在区域内停留超过3秒
4. 控制台会打印报警信息
5. 自动开始录制视频

### 4. 查看录像文件

在Web界面中：
1. 点击 **录制查看** Tab
2. 查看保存的视频文件列表
3. 点击 **打开文件位置** 查看MP4文件

录像文件命名格式：`clip_20260415-163000_detection.mp4`

---

## 配置说明

`config.ini` 主要配置项：

```ini
[general]
default_conf_threshold = 0.5    # 检测置信度阈值
default_model_variant = yolo11m # 模型版本

[input]
use_webcam = true               # 使用USB摄像头
webcam_index = 0                # 摄像头编号

[zones]
enabled = true                  # 启用区域检测
clip_duration = 10              # 录制时长（秒）
polygons = gate=45,67;41,85;69,86;64,67
                # 区域定义：名称=坐标点（百分比）

[dingtalk]
enabled = false                 # 钉钉推送开关
webhook_url =                   # 钉钉机器人地址
```

### 区域坐标说明

区域使用百分比坐标，自动适配分辨率：

```
polygons = gate=45,67;41,85;69,86;64,67

含义：
- gate：区域名称
- 45,67：第一个点的位置（画面宽度45%，高度67%）
- 41,85：第二个点的位置
- 多个点用分号分隔
- 多个区域用竖线分隔：gate=... | door=...
```

---

## 项目结构

```
dvr-yolov8-detection-custom/
├── yolov8_live_rtmp_stream_detection.py  # 主程序入口
├── zone_manager.py                        # 区域检测模块
├── video_recorder.py                      # 视频录制模块
├── nicegui_dashboard.py                   # Web管理界面
├── dingtalk_notify.py                     # 钉钉推送模块
├── daily_report.py                        # 每日统计模块
├── config.ini                             # 配置文件
├── start.bat                              # Windows启动脚本
├── requirements.txt                       # Python依赖
└── README.md                              # 说明文档
```

---

## 常见问题

### Q1：摄像头无法打开

检查摄像头编号：
```ini
[input]
webcam_index = 0    # 尝试改为1、2...
```

### Q2：GPU加速不工作

确保安装了CUDA版本的PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q3：FPS很低

尝试使用更轻量的模型：
```ini
[general]
default_model_variant = yolo11n    # nano版本，速度更快
```

### Q4：Web界面无法打开

检查端口5001是否被占用，或修改端口：
```python
# nicegui_dashboard.py 中修改
ui.run(port=5002)
```

---

## 开发团队

本项目为毕业设计课程作品，由7人团队共同完成。

---

## 致谢

基于 [FlyingFathead/dvr-yolov8-detection](https://github.com/FlyingFathead/dvr-yolov8-detection) 开源项目二次开发。

技术栈：
- [ultralytics](https://github.com/ultralytics/ultralytics) — YOLO模型
- [supervision](https://github.com/roboflow/supervision) — 检测辅助
- [nicegui](https://github.com/zauberzeug/nicegui) — Web界面
- [opencv](https://github.com/opencv/opencv) — 视频处理

---

## License

MIT License