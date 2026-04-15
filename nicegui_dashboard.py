#!/usr/bin/env python3
"""
Surveillance Dashboard (NiceGUI)
Port: 5001
Features:
  - 实时视频显示 + 点击添加区域
  - 检测记录实时列表
  - 区域管理（添加/删除）
  - 系统信息 + 录像文件
"""
import asyncio
import threading
import time
import cv2
import base64
from datetime import datetime
import os
import configparser
from datetime import datetime
from nicegui import ui

# ============================================================
# 全局状态（由主程序调用更新）
# ============================================================
_frame = None
_frame_lock = threading.Lock()
_detections = []
_detections_lock = threading.Lock()
_zone_stats = {}
_fps = 0.0
_total_detections = 0
_config = {}

# 区域编辑状态
_editing_points = []
_editing_mode = False  # 编辑模式开关
_frame_width = 640  # 实际帧宽度
_frame_height = 480  # 实际帧高度

# 全局刷新触发器
_refresh_trigger = 0  # 每次有新检测时增加
_last_refresh_count = 0  # 上次刷新时的计数

# Track 时间追踪
_track_times = {}  # {track_id: {'enter': time, 'exit': time, 'photo': path}}
_seen_track_ids = set()  # 已记录的 track ID


def set_frame(frame):
    """更新视频帧"""
    global _frame
    with _frame_lock:
        _frame = frame.copy() if frame is not None else None


# 全局刷新触发器
_refresh_trigger = 0  # 每次有新检测时增加

def add_detection(info):
    """添加检测 - 按 track_id 去重，记录进出时间"""
    global _detections, _total_detections, _seen_track_ids, _track_times, _refresh_trigger
    
    # 转换 track_id 为 Python int
    track_id_raw = info.get('track_id')
    track_id = int(track_id_raw) if track_id_raw is not None else None
    
    # 时间处理 - 直接提取时间部分
    timestamp_full = info.get('timestamp', '')
    if timestamp_full and len(timestamp_full) > 10:
        timestamp = timestamp_full.split(' ')[1] if ' ' in timestamp_full else timestamp_full[-8:]
    else:
        timestamp = datetime.now().strftime('%H:%M:%S')
    
    is_new = False  # 是否是新检测
    
    with _detections_lock:
        if track_id is not None and track_id in _seen_track_ids:
            if track_id in _track_times:
                _track_times[track_id]['exit'] = timestamp
            return
        
        if track_id is not None:
            _seen_track_ids.add(track_id)
            is_new = True
            
            images = info.get('image_filenames', {})
            photo_path = ''
            if images and 'detection_area' in images:
                photo_path = f"E:/yolo_detections/{images.get('detection_area', '')}"
            
            named_zones = info.get('named_zones', [])
            zone_name = ', '.join(named_zones) if named_zones else ''
            
            _track_times[track_id] = {
                'enter': timestamp,
                'exit': '',
                'photo': photo_path,
                'conf': float(info.get('confidence', 0)),
                'zone': zone_name
            }
        
        _detections.insert(0, info)
        _detections = _detections[:100]
        _total_detections += 1
    
    # 触发 UI 刷新（新检测时）
    if is_new:
        _refresh_trigger += 1


def update_zone_stats(stats):
    """更新区域统计"""
    global _zone_stats
    _zone_stats = stats if stats else {}


def update_fps(fps_val):
    """更新 FPS"""
    global _fps
    _fps = fps_val


def update_config(cfg):
    """更新配置信息"""
    global _config
    _config = cfg if cfg else {}


# ============================================================
# 区域配置辅助函数
# ============================================================
def _load_zone_config():
    """加载区域配置 - 返回 dict 兼容 ConfigParser"""
    config = configparser.ConfigParser(interpolation=None)
    config.read('config.ini', encoding='utf-8')
    return config


def _save_zone_config(polygons_str):
    """保存区域配置 - 保持原文件格式不变"""
    with open('config.ini', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find [zones] section and update polygons line
    in_zones = False
    new_lines = []
    polygons_updated = False
    for line in lines:
        stripped = line.strip()
        if stripped == '[zones]':
            in_zones = True
            new_lines.append(line)
            continue
        if in_zones:
            if stripped.startswith('['):
                in_zones = False
            elif stripped.startswith('polygons'):
                new_lines.append(f'polygons = {polygons_str}\n')
                polygons_updated = True
                continue
        new_lines.append(line)
    
    # If polygons line wasn't found in [zones], add it
    if not polygons_updated:
        for i, line in enumerate(new_lines):
            if line.strip() == '[zones]':
                new_lines.insert(i + 1, f'polygons = {polygons_str}\n')
                break
    
    with open('config.ini', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def _parse_polygons(polygons_data):
    """解析区域字符串为列表 - 支持 ConfigParser、dict 或 str 输入"""
    # 如果输入是 ConfigParser，取 polygons 字段
    if isinstance(polygons_data, configparser.ConfigParser):
        polygons_str = polygons_data.get('zones', 'polygons', fallback='')
    elif isinstance(polygons_data, dict):
        polygons_str = polygons_data.get('polygons', '')
    else:
        polygons_str = polygons_data or ''
    
    result = []
    if not polygons_str:
        return result
    for item in polygons_str.split('|'):
        item = item.strip()
        if not item:
            continue
        eq = item.find('=')
        if eq < 0:
            continue
        name = item[:eq].strip()
        coords = item[eq + 1:].strip()
        pts = []
        for pair in coords.split(';'):
            parts = pair.strip().split(',')
            if len(parts) == 2:
                try:
                    pts.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
        if len(pts) >= 3:
            result.append({'name': name, 'points': pts})
    return result


# ============================================================
# NiceGUI 主页面
# ============================================================
@ui.page('/')
async def main_page():
    # 暗色主题
    ui.dark_mode().enable()
    
    # 自定义样式
    ui.add_head_html('''<style>
        body { background: #0d1117; margin: 0; }
        .nicegui-content { padding: 4px; max-width: 100%; }
        .q-card { background: #161b22; border: 1px solid #30363d; color: #c9d1d9; }
        .q-tab { color: #8b949e; font-size: 13px; }
        .q-tab--active { color: #58a6ff; }
        .det-item { 
            padding: 8px 12px; 
            border-bottom: 1px solid #21262d; 
            font-size: 12px; 
            font-family: Consolas, monospace;
        }
        .zone-item {
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 8px;
        }
    </style>''')

    # 顶部栏
    with ui.header().classes('items-center justify-between px-4').style(
        'background:#161b22;border-bottom:1px solid #30363d;height:40px;'
    ):
        ui.icon('videocam', size='sm').style('color:#58a6ff;')
        ui.label('监控系统').style('font-size:16px;font-weight:bold;color:#58a6ff;margin-left:8px;')
        
        with ui.row().classes('gap-4'):
            fps_label = ui.label('FPS: --').style('color:#8b949e;font-size:12px;')
            total_label = ui.label('检测: 0').style('color:#8b949e;font-size:12px;')
            clock_label = ui.label('--:--:--').style('color:#8b949e;font-size:12px;')

    # 页面存活标志
    page_alive = True

    def on_disconnect():
        page_alive = False

    ui.context.client.on_disconnect(on_disconnect)

    # 定时更新顶部栏
    def update_header():
        if not page_alive:
            return
        try:
            clock_label.text = datetime.now().strftime('%H:%M:%S')
            fps_label.text = f'FPS: {_fps:.0f}'
            total_label.text = f'检测: {_total_detections}'
        except:
            pass
    ui.timer(1.0, update_header)

    # ===== 主布局 =====
    with ui.splitter(value=55).classes('w-full').style('height:calc(100vh - 40px);') as splitter:
        # 左侧：视频
        with splitter.before:
            with ui.column().classes('w-full p-2'):
                # 视频显示 - 固定尺寸与帧一致 (640x480)
                video_img = ui.interactive_image(
                    source='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
                ).style('width:640px;height:480px;border-radius:4px;border:1px solid #30363d;cursor:crosshair;object-fit:contain;background:#000;')
                
                # 绑定点击事件
                video_img.on('click', handle_video_click)
                
                # 提示栏
                hint_label = ui.label('💡 点击"编辑"按钮开始').style(
                    'color:#8b949e;font-size:11px;margin-top:4px;'
                )

            # 视频更新任务
            async def update_video():
                global _frame_width, _frame_height
                while page_alive:
                    try:
                        with _frame_lock:
                            if _frame is not None:
                                display = _frame.copy()
                                # 保存帧尺寸
                                _frame_height, _frame_width = display.shape[:2]
                                
                                # 绘制编辑点
                                if _editing_points:
                                    h, w = display.shape[:2]
                                    pts = [(int(p[0]/100*w), int(p[1]/100*h)) for p in _editing_points]
                                    
                                    # 绘制连线
                                    for i in range(1, len(pts)):
                                        cv2.line(display, pts[i-1], pts[i], (0, 255, 0), 2)
                                    # 闭合预览
                                    if len(pts) >= 3:
                                        cv2.line(display, pts[-1], pts[0], (0, 255, 0), 1)
                                    
                                    # 绘制点
                                    for i, pt in enumerate(pts):
                                        cv2.circle(display, pt, 8, (0, 255, 0), -1)
                                        cv2.putText(display, str(i+1), (pt[0]-5, pt[1]+5), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                
                                _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                b64 = base64.b64encode(buf).decode()
                                video_img.source = f'data:image/jpeg;base64,{b64}'
                                
                                # 动态更新显示尺寸与帧尺寸一致
                                video_img.style(f'width:{_frame_width}px;height:{_frame_height}px;object-fit:none;')
                    except Exception as e:
                        print(f"[Video Error] {e}")
                    await asyncio.sleep(0.08)  # ~12 FPS
            asyncio.create_task(update_video())

        # 右侧：标签页
        with splitter.after:
            with ui.tabs().classes('w-full').style('background:#161b22;') as tabs:
                tab_det = ui.tab('检测记录')
                tab_zone = ui.tab('区域管理')
                tab_clips = ui.tab('录制查看')
                tab_sys = ui.tab('系统信息')

            with ui.tab_panels(tabs, value=tab_det).classes('w-full').style('background:#0d1117;'):
                
                # ===== 检测记录 =====
                with ui.tab_panel(tab_det):
                    ui.label('📊 检测记录').style('font-size:14px;font-weight:bold;color:#58a6ff;margin:10px')
                    
                    with ui.row().classes('gap-4').style('margin:10px'):
                        ui.label(f'人数: {len(_seen_track_ids)}').style('color:#3fb950;font-size:12px;font-weight:bold')
                        ui.label(f'闯入: {sum(1 for t in _track_times.values() if t.get("zone"))}').style('color:#f85149;font-size:12px')
                    
                    ui.separator().style('background:#30363d;margin:5px 10px')
                    
                    det_list = ui.column().classes('w-full p-2 gap-2')
                    
                    def refresh_detections():
                        if not page_alive:
                            return
                        det_list.clear()
                        with det_list:
                            # 显示 track 记录（不重复）
                            if not _track_times:
                                ui.label('等待检测...').style('color:#8b949e;padding:20px;text-align:center')
                            else:
                                for tid, data in list(_track_times.items())[-20:]:
                                    enter = data.get('enter', '--')
                                    exit = data.get('exit', '')
                                    photo = data.get('photo', '')
                                    conf = data.get('conf', 0)
                                    zone = data.get('zone', '')
                                    
                                    # 计算停留时间
                                    duration = ''
                                    if enter and exit:
                                        try:
                                            e = datetime.strptime(enter, '%H:%M:%S')
                                            x = datetime.strptime(exit, '%H:%M:%S')
                                            secs = int((x - e).total_seconds())
                                            if secs >= 60:
                                                duration = f'{secs//60}分{secs%60}秒'
                                            else:
                                                duration = f'{secs}秒'
                                        except:
                                            pass
                                    
                                    bg = '#f8514920' if zone else '#21262d'
                                    
                                    with ui.card().classes('w-full').style(f'padding:8px;background:{bg};border-radius:6px;margin:2px'):
                                        with ui.row().classes('w-full items-start gap-2'):
                                            # 照片（可点击放大）
                                            if photo and os.path.exists(photo):
                                                img_btn = ui.button(on_click=lambda p=photo: show_photo_dialog(p)).style('padding:0;border:none;background:transparent')
                                                with img_btn:
                                                    ui.image(photo).style('width:100px;height:75px;border-radius:4px;object-fit:cover')
                                            
                                            # 信息
                                            with ui.column().classes('flex-1'):
                                                with ui.row().classes('items-center gap-2'):
                                                    ui.label(f'#{tid}').style('color:#c9d1d9;font-weight:bold;font-size:13px')
                                                    ui.label(f'{conf:.0%}').style('color:#3fb950;font-size:11px')
                                                    if zone:
                                                        ui.label(f'⚠️{zone}').style('color:#f85149;font-size:11px')
                                                
                                                with ui.row().classes('gap-2 mt-1'):
                                                    ui.label(f'进入:{enter}').style('color:#8b949e;font-size:10px')
                                                    if exit:
                                                        ui.label(f'离开:{exit}').style('color:#8b949e;font-size:10px')
                                                    if duration:
                                                        ui.label(f'停留:{duration}').style('color:#d29922;font-size:10px')
                    
                    # 照片弹窗函数
                    def show_photo_dialog(photo_path):
                        with ui.dialog() as dialog:
                            with ui.card().style('padding:10px;background:#161b22'):
                                ui.label('📷 人物截图').style('color:#58a6ff;font-size:14px;margin-bottom:8px')
                                ui.image(photo_path).style('width:500px;max-height:400px;border-radius:4px')
                                ui.button('关闭', on_click=dialog.close).style('background:#30363d;color:#fff;margin-top:8px')
                            dialog.open()
                    
                    ui.timer(5.0, refresh_detections)  # 降低刷新频率到5秒
                    refresh_detections()

                # ===== 区域管理 =====
                with ui.tab_panel(tab_zone):
                    ui.label('🎯 区域管理').style('font-size:14px;font-weight:bold;color:#58a6ff;margin:10px')
                    
                    # 编辑按钮和状态
                    with ui.row().classes('gap-4 items-center').style('margin:10px'):
                        status_label = ui.label('观看模式').style('color:#8b949e;font-size:12px')
                        edit_btn = ui.button('✏️ 编辑', on_click=toggle_edit_mode).props('unelevated dense')
                        edit_btn.style('background:#1f6feb;color:#fff')
                    
                    def update_status():
                        global _editing_mode
                        if _editing_mode:
                            status_label.text = '✏️ 编辑中'
                            status_label.style('color:#3fb950')
                            edit_btn.text = '❌ 退出'
                            edit_btn.style('background:#da3633;color:#fff')
                        else:
                            status_label.text = '👁 观看模式'
                            status_label.style('color:#8b949e')
                            edit_btn.text = '✏️ 编辑'
                            edit_btn.style('background:#1f6feb;color:#fff')
                    ui.timer(0.5, update_status)
                    
                    # 编辑面板（3点后显示）
                    edit_panel = ui.column().classes('w-full p-2').style('display:none;')
                    with edit_panel:
                        ui.label('点数').style('color:#d29922')
                        name_input = ui.input('名称', value='').props('dense outlined')
                        ui.button('保存', on_click=lambda: save_zone_action(name_input)).style('background:#238636;color:#fff')
                        ui.button('清除', on_click=clear_points_action).style('background:#da3633;color:#fff')
                    
                    def update_edit():
                        edit_panel.style('display:block' if _editing_mode and len(_editing_points) >= 3 else 'display:none')
                    ui.timer(0.5, update_edit)
                    
                    # 已有区域
                    ui.label('已配置区域').style('font-size:12px;color:#58a6ff;margin:10px')
                    zone_list = ui.column().classes('w-full p-2')
                    
                    def refresh_zones():
                        if not page_alive:
                            return
                        zone_list.clear()
                        with zone_list:
                            cfg = _load_zone_config()
                            zones = _parse_polygons(cfg)
                            if not zones:
                                ui.label('暂无区域').style('color:#8b949e;padding:8px')
                            else:
                                # 最新添加的在最上面
                                for i in range(len(zones) - 1, -1, -1):
                                    z = zones[i]
                                    with ui.row().classes('w-full items-center justify-between').style('padding:8px;background:#21262d;border-radius:4px;margin-bottom:4px;'):
                                        ui.label(f'#{i+1} {z["name"]}').style('color:#c9d1d9;font-size:12px')
                                        ui.button('删除', on_click=lambda idx=i: delete_zone_action(idx)).props('dense flat').style('color:#f85149')
                    ui.timer(5.0, refresh_zones)
                    refresh_zones()

                # ===== 录制查看 =====
                with ui.tab_panel(tab_clips):
                    ui.label('🎬 录制查看').style('font-size:14px;font-weight:bold;color:#58a6ff;margin:10px')
                    
                    clips_container = ui.column().classes('w-full p-2')
                    
                    def refresh_clips():
                        if not page_alive:
                            return
                        clips_container.clear()
                        with clips_container:
                            save_dir = 'E:/yolo_detections/'
                            clips = []
                            if os.path.exists(save_dir):
                                for root, dirs, files in os.walk(save_dir):
                                    for f in files:
                                        if f.endswith('.mp4'):
                                            full_path = os.path.join(root, f)
                                            mtime = os.path.getmtime(full_path)
                                            clips.append((mtime, full_path, f))
                            clips.sort(reverse=True)  # 最新在前
                            
                            if not clips:
                                ui.label('暂无录像').style('color:#8b949e;padding:8px')
                            else:
                                ui.label(f'共 {len(clips)} 个录像').style('color:#8b949e;font-size:11px;margin-bottom:8px')
                                # 只显示最近20个
                                for mtime, full_path, fname in clips[:20]:
                                    from datetime import datetime as _dt
                                    time_str = _dt.fromtimestamp(mtime).strftime('%m-%d %H:%M:%S')
                                    size_mb = os.path.getsize(full_path) / 1024 / 1024
                                    with ui.row().classes('w-full items-center justify-between').style(
                                        'padding:6px 10px;background:#21262d;border-radius:4px;margin-bottom:3px;'
                                    ):
                                        with ui.column().style('gap:0'):
                                            ui.label(fname).style('color:#c9d1d9;font-size:11px')
                                            ui.label(f'{time_str}  {size_mb:.1f}MB').style('color:#8b949e;font-size:10px')
                                        ui.button('📂', on_click=lambda p=full_path: os.startfile(p)).props(
                                            'dense flat').style('color:#58a6ff').tooltip('打开文件位置')
                    
                    ui.timer(10.0, refresh_clips)
                    refresh_clips()

                # ===== 系统信息 =====
                with ui.tab_panel(tab_sys):
                    ui.label('⚙️ 系统信息').style('font-size:14px;font-weight:bold;color:#58a6ff;margin:10px')
                    
                    # 状态卡片
                    with ui.card().classes('w-full').style('padding:10px;background:#161b22;margin:10px'):
                        ui.label('运行状态').style('font-size:12px;color:#58a6ff;margin-bottom:8px')
                        
                        fps_row = ui.row().classes('gap-4')
                        with fps_row:
                            ui.icon('speed', size='sm').style('color:#3fb950')
                            sys_fps = ui.label(f'FPS: {_fps:.1f}').style('color:#3fb950;font-size:12px')
                        
                        det_row = ui.row().classes('gap-4 mt-2')
                        with det_row:
                            ui.icon('visibility', size='sm').style('color:#58a6ff')
                            sys_det = ui.label(f'检测: {_total_detections}').style('color:#58a6ff;font-size:12px')
                        
                        zone_row = ui.row().classes('gap-4 mt-2')
                        with zone_row:
                            ui.icon('crop_free', size='sm').style('color:#d29922')
                            zones = _parse_polygons(_load_zone_config())
                            ui.label(f'区域: {len(zones)}').style('color:#d29922;font-size:12px')
                        
                        gpu_row = ui.row().classes('gap-4 mt-2')
                        with gpu_row:
                            ui.icon('memory', size='sm').style('color:#8b949e')
                            ui.label('GPU: RTX 3050 Ti').style('color:#8b949e;font-size:12px')
                    
                    # 录像统计
                    with ui.card().classes('w-full').style('padding:10px;background:#161b22;margin:10px'):
                        ui.label('录像统计').style('font-size:12px;color:#58a6ff;margin-bottom:8px')
                        
                        # 统计录像文件
                        clips_count = 0
                        save_dir = 'E:/yolo_detections/'
                        if os.path.exists(save_dir):
                            for root, dirs, files in os.walk(save_dir):
                                clips_count += len([f for f in files if f.endswith('.mp4')])
                        
                        clips_row = ui.row().classes('gap-4')
                        with clips_row:
                            ui.icon('videocam', size='sm').style('color:#3fb950')
                            ui.label(f'录像: {clips_count} 个').style('color:#3fb950;font-size:12px')
                    
                    def update_sys():
                        sys_fps.text = f'FPS: {_fps:.1f}'
                        sys_det.text = f'检测: {_total_detections}'
                    ui.timer(2.0, update_sys)


# ============ Zone Editor Functions ============

# ============================================================
# 区域编辑函数
# ============================================================
def toggle_edit_mode():
    """切换编辑模式"""
    global _editing_mode, _editing_points
    _editing_mode = not _editing_mode
    if not _editing_mode:
        _editing_points.clear()
    ui.notify(f'编辑模式: {"开启" if _editing_mode else "关闭"}', type='info', timeout=1.5)

def handle_video_click(e):
    """处理视频点击 - 只在编辑模式下响应"""
    global _editing_points, _frame_width, _frame_height
    if not _editing_mode:
        return
    
    try:
        args = e.args or {}
        x_px = args.get('offsetX', 0)
        y_px = args.get('offsetY', 0)
        
        # 使用实际帧尺寸计算百分比
        x_pct = round(x_px / _frame_width * 100, 1)
        y_pct = round(y_px / _frame_height * 100, 1)
        
        print(f"[Click] px=({x_px},{y_px}) frame=({_frame_width}x{_frame_height}) pct=({x_pct}%,{y_pct}%)")
        
        if x_pct <= 0 or y_pct <= 0 or x_pct > 100 or y_pct > 100:
            ui.notify('坐标超出范围', type='warning', timeout=1)
            return
        
        _editing_points.append((x_pct, y_pct))
        ui.notify(f'点 #{len(_editing_points)}: ({x_pct:.1f}%, {y_pct:.1f}%)', type='positive', timeout=1)
    except Exception as ex:
        print(f"[点击错误] {ex}")

def save_zone_action(name_input):
    """保存区域"""
    global _editing_points
    name = name_input.value.strip()
    if not name:
        ui.notify('请输入名称', type='warning')
        return
    if len(_editing_points) < 3:
        ui.notify(f'需要至少3个点（当前{len(_editing_points)}个）', type='warning')
        return
    
    coords = ';'.join([f'{int(p[0])},{int(p[1])}' for p in _editing_points])
    cfg = _load_zone_config()
    existing = cfg.get('zones', 'polygons', fallback='')
    new_val = f"{existing} | {name}={coords}" if existing else f"{name}={coords}"
    _save_zone_config(new_val)
    _editing_points.clear()
    name_input.value = ''
    # Hot-reload zone_mgr
    _reload_zones()
    ui.notify(f'区域 "{name}" 已保存并生效！', type='positive', timeout=2)

def clear_points_action():
    """清除所有点"""
    global _editing_points
    _editing_points.clear()
    ui.notify('已清除', type='info', timeout=1)

def delete_zone_action(index):
    """删除区域"""
    cfg = _load_zone_config()
    zones = _parse_polygons(cfg)
    if 0 <= index < len(zones):
        removed = zones.pop(index)
        new_val = ' | '.join([f'{z["name"]}={";".join([f"{int(p[0])},{int(p[1])}" for p in z["points"]])}' for z in zones])
        _save_zone_config(new_val)
        # Hot-reload zone_mgr if available
        _reload_zones()
        ui.notify(f'区域 "{removed["name"]}" 已删除并生效！', type='positive', timeout=2)

def _reload_zones():
    """Hot-reload zone manager"""
    try:
        import yolov8_live_rtmp_stream_detection as main_mod
        zone_mgr = getattr(main_mod, 'zone_mgr', None)
        if zone_mgr is not None:
            zone_mgr.reload(config=_load_zone_config())
    except Exception as e:
        print(f"[Dashboard] zone reload failed: {e}")


# ============================================================
# 启动函数
# ============================================================
def start_dashboard(host='127.0.0.1', port=5001):
    """启动 Dashboard"""
    import webbrowser
    
    url = f'http://{host}:{port}'
    
    # 延迟打开浏览器
    def open_browser():
        time.sleep(2.5)
        try:
            webbrowser.open(url)
            print(f"[Dashboard] 浏览器已打开: {url}")
        except Exception as e:
            print(f"[Dashboard] 无法打开浏览器: {e}")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    print(f"[Dashboard] 启动中... 地址: {url}")
    ui.run(host=host, port=port, reload=False, show=False, title='监控系统', native=False)