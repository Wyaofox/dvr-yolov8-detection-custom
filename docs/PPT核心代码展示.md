# PPT 核心代码展示（带逐行注解）

---

## 代码段一：Bbox-Polygon 重叠度计算（zone_manager.py）

```python
def _box_polygon_overlap(box, polygon):
    """计算检测框与禁区多边形的重叠比例，返回 0.0~1.0"""
    x1, y1, x2, y2 = box                    # 解包检测框坐标（左上角+右下角）
    box_area = (x2 - x1) * (y2 - y1)        # 计算检测框面积（宽×高）

    h, w = int(y2) + 1, int(x2) + 1          # 创建mask画布的尺寸（取框的右下角）
    mask = np.zeros((h, w), dtype=np.uint8)   # 创建全黑画布（全0矩阵）

    cv2.fillPoly(mask, [polygon], 1)          # 把禁区多边形涂白（值为1）

    box_mask = np.zeros_like(mask)            # 再创建一个同样大小的全黑画布
    box_mask[y1:y2, x1:x2] = 1               # 把检测框区域涂白（值为1）

    intersection = np.logical_and(mask, box_mask).sum()  # 两个mask做"与"运算，求交集像素数
    return float(intersection) / float(box_area)          # 交集面积 ÷ 框面积 = 重叠比例
```

### 讲解要点
- 把"几何问题"变成"像素运算"——简单直观
- fillPoly 一次性渲染整个多边形，非常高效
- logical_and 就是像素级别的"同时为1"判断
- 最终返回 0.0~1.0 之间的比例，≥0.3 就算人在禁区内

---

## 代码段二：停留计时 + 告警触发（zone_manager.py update方法）

```python
for i in range(len(detections)):                # 遍历当前帧检测到的每个人
    bbox = detections.xyxy[i]                   # 取第i个人的检测框坐标

    if self._bbox_in_zone(bbox, polygon):       # 判断这个人是否在禁区内（重叠≥30%）
        tid = tracker_ids[i]                    # 取这个人的跟踪ID（ByteTrack分配的）
        active_tracks.add(tid)                  # 记录当前在禁区内的跟踪ID

        if tid not in dwell_start[name]:        # 如果这个人之前没进过这个禁区
            dwell_start[name][tid] = now        # 记录他首次进入的时间戳

        dwell = now - dwell_start[name][tid]    # 计算他已经停留了多少秒

        if dwell >= 3.0 and (name, tid) not in alerted:  # 停留≥3秒 且 没告警过
            alerted.add((name, tid))            # 标记为已告警（防止重复）
            zone_alerts.append((name, tid))     # 产生一条告警记录
```

### 讲解要点
- 用 tracker_id 区分不同的人，互不影响
- dwell_start 字典记录每个人的首次进入时间
- 3秒内离开 → 计时归零（后续清理逻辑）
- _alerted 集合防止同一个人重复告警

---

## 代码段三：GPU 模型加载（主脚本）

```python
model = YOLO("yolo11m.pt")                     # 加载 YOLOv11m 预训练模型（自动下载）

if torch.cuda.is_available():                  # 检测是否有 NVIDIA GPU 可用
    model.to("cuda:0")                         # 把模型搬到 GPU 上（加速推理）
else:
    model.to("cpu")                            # 没有GPU就用CPU（会慢很多）

results = model(frame, conf=0.5)               # 对视频帧做推理，置信度阈值50%
detections = sv.Detections.from_ultralytics(results)  # 转成 supervision 格式方便后续处理
```

### 讲解要点
- YOLO("yolo11m.pt") 一行代码加载完整模型
- model.to("cuda:0") 是关键——把计算从CPU搬到GPU
- conf=0.5 表示只保留置信度≥50%的检测结果
- sv.Detections.from_ultralytics 做格式转换，衔接检测和可视化
- GPU上单帧推理约40ms，CPU上约300ms，差距7倍
