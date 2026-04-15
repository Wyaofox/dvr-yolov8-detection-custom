#!/usr/bin/env python3
"""
轻量中文监控面板 - 替代原有 Web UI
- 低延迟：WebSocket 推送 + 压缩缩略图
- 中文界面
- 实时检测列表 + 统计
"""

import cv2
import threading
import time
import json
import base64
import webbrowser
import numpy as np
from flask import Flask, render_template_string, jsonify, Response, request
from waitress import serve

app = Flask(__name__)

# Shared state (set from main detection script)
_output_frame = None
_frame_lock = threading.Lock()
_detections_list = []  # recent detections
_detections_lock = threading.Lock()
_detection_count = 0
_fps = 0.0
_config = {}

# ============ HTML Template (Chinese Dashboard) ============

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>实时监控面板</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: "Microsoft YaHei","PingFang SC",sans-serif; background:#0d1117; color:#c9d1d9; }
.header { background:#161b22; padding:12px 24px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #30363d; }
.header h1 { font-size:18px; color:#58a6ff; }
.status-bar { display:flex; gap:20px; font-size:13px; }
.status-item { display:flex; align-items:center; gap:6px; }
.dot { width:8px; height:8px; border-radius:50%; }
.dot.green { background:#3fb950; animation:pulse 2s infinite; }
.dot.red { background:#f85149; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

.container { display:grid; grid-template-columns:1fr 380px; height:calc(100vh - 52px); }

.video-panel { padding:16px; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.video-panel img { max-width:100%; max-height:calc(100vh - 100px); border-radius:8px; border:1px solid #30363d; }

.side-panel { background:#161b22; border-left:1px solid #30363d; display:flex; flex-direction:column; overflow:hidden; }
.tab-bar { display:flex; border-bottom:1px solid #30363d; }
.tab { flex:1; padding:10px; text-align:center; cursor:pointer; font-size:13px; color:#8b949e; transition:.2s; }
.tab.active { color:#58a6ff; border-bottom:2px solid #58a6ff; background:#0d1117; }
.tab:hover { color:#c9d1d9; }

.tab-content { flex:1; overflow-y:auto; padding:12px; }
.tab-content::-webkit-scrollbar { width:4px; }
.tab-content::-webkit-scrollbar-thumb { background:#30363d; border-radius:2px; }

.det-card { background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:12px; margin-bottom:10px; transition:.2s; }
.det-card:hover { border-color:#58a6ff; }
.det-card .time { font-size:11px; color:#8b949e; }
.det-card .info { display:flex; justify-content:space-between; align-items:center; margin-top:6px; }
.det-card .conf { font-size:14px; font-weight:bold; }
.conf.high { color:#3fb950; }
.conf.mid { color:#d29922; }
.conf.low { color:#f85149; }
.det-card .coords { font-size:11px; color:#8b949e; }
.det-card .zone { font-size:11px; color:#58a6ff; background:#1f2937; padding:2px 8px; border-radius:10px; }

.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px; }
.stat-box { background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:16px; text-align:center; }
.stat-box .value { font-size:28px; font-weight:bold; color:#58a6ff; }
.stat-box .label { font-size:12px; color:#8b949e; margin-top:4px; }

.setting-row { display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #21262d; }
.setting-row .name { font-size:13px; }
.setting-row .val { font-size:13px; color:#58a6ff; font-family:monospace; }

.empty-state { text-align:center; padding:40px 20px; color:#484f58; }
.empty-state .icon { font-size:48px; margin-bottom:12px; }

#detection-list { }
</style>
</head>
<body>

<div class="header">
  <h1>📹 实时监控面板</h1>
  <div class="status-bar">
    <div class="status-item"><div class="dot green" id="status-dot"></div><span id="status-text">运行中</span></div>
    <div class="status-item">🖥 FPS: <strong id="fps-val">0</strong></div>
    <div class="status-item">🎯 检测数: <strong id="det-count">0</strong></div>
    <div class="status-item" id="clock"></div>
  </div>
</div>

<div class="container">
  <div class="video-panel">
    <img id="video" src="/video_feed" alt="实时画面" onerror="this.style.display='none';document.getElementById('no-video').style.display='block'">
    <div id="no-video" class="empty-state" style="display:none;">
      <div class="icon">📹</div>
      <div>等待视频画面...</div>
    </div>
  </div>

  <div class="side-panel">
    <div class="tab-bar">
      <div class="tab active" onclick="switchTab('detections')">检测记录</div>
      <div class="tab" onclick="switchTab('persons')">人员档案</div>
      <div class="tab" onclick="switchTab('zones')">区域设置</div>
      <div class="tab" onclick="switchTab('stats')">统计信息</div>
      <div class="tab" onclick="switchTab('settings')">系统设置</div>
    </div>

    <div class="tab-content" id="tab-detections">
      <div id="detection-list"></div>
    </div>

    <div class="tab-content" id="tab-persons" style="display:none;">
      <div id="persons-list"></div>
    </div>

    <div class="tab-content" id="tab-zones" style="display:none;">
      <div style="margin-bottom:12px;">
        <h3 style="font-size:14px;color:#58a6ff;margin-bottom:8px;">Region Editor</h3>
        <p style="font-size:12px;color:#8b949e;margin-bottom:10px;">Click on the snapshot to add polygon points. Switch to Line mode to add line endpoints.</p>
        <div style="display:flex;gap:8px;margin-bottom:10px;">
          <button onclick="zoneMode='polygon'" id="btn-polygon" style="padding:6px 14px;border:1px solid #30363d;background:#0d1117;color:#c9d1d9;border-radius:4px;cursor:pointer;">Polygon Zone</button>
          <button onclick="zoneMode='line'" id="btn-line" style="padding:6px 14px;border:1px solid #30363d;background:#0d1117;color:#c9d1d9;border-radius:4px;cursor:pointer;">Counting Line</button>
          <button onclick="undoPoint()" style="padding:6px 14px;border:1px solid #30363d;background:#0d1117;color:#c9d1d9;border-radius:4px;cursor:pointer;">Undo</button>
          <button onclick="clearPoints()" style="padding:6px 14px;border:1px solid #30363d;background:#0d1117;color:#c9d1d9;border-radius:4px;cursor:pointer;">Clear</button>
          <button onclick="saveZones()" style="padding:6px 14px;border:1px solid #58a6ff;background:#1f6feb;color:#fff;border-radius:4px;cursor:pointer;">Save to Config</button>
        </div>
        <div style="display:flex;gap:8px;margin-bottom:10px;">
          <input id="zone-name" placeholder="Zone name (e.g. entrance)" style="flex:1;padding:6px 10px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;border-radius:4px;">
        </div>
      </div>
      <div style="position:relative;display:inline-block;">
        <canvas id="zone-canvas" style="border:1px solid #30363d;border-radius:8px;cursor:crosshair;"></canvas>
      </div>
      <div id="zone-current" style="margin-top:10px;font-size:12px;color:#8b949e;"></div>
    </div>

    <div class="tab-content" id="tab-stats" style="display:none;">
      <div class="stat-grid">
        <div class="stat-box"><div class="value" id="stat-total">0</div><div class="label">总检测数</div></div>
        <div class="stat-box"><div class="value" id="stat-fps">0</div><div class="label">当前帧率</div></div>
        <div class="stat-box"><div class="value" id="stat-conf">-</div><div class="label">平均置信度</div></div>
        <div class="stat-box"><div class="value" id="stat-recent">0</div><div class="label">近5分钟检测</div></div>
      </div>
    </div>

    <div class="tab-content" id="tab-settings" style="display:none;">
      <div id="settings-list"></div>
    </div>
  </div>
</div>

<script>
let detCount = 0;
let allDetections = [];

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
  event.target.classList.add('active');
  document.getElementById('tab-' + name).style.display = 'block';
}

function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent = now.toLocaleTimeString('zh-CN');
}
setInterval(updateClock, 1000);
updateClock();

// Poll detections
let _totalDetCount = 0;
async function fetchDetections() {
  try {
    const resp = await fetch('/api/detections');
    const data = await resp.json();
    if (data.detections && data.detections.length > 0) {
      allDetections = data.detections;
      renderDetections(data.detections.slice(0, 50));
      _totalDetCount = data.detections.length;
      document.getElementById('det-count').textContent = _totalDetCount;
      document.getElementById('stat-total').textContent = _totalDetCount;
    }
  } catch(e) {}
}

function renderDetections(dets) {
  const list = document.getElementById('detection-list');
  let html = '';
  dets.forEach(d => {
    const conf = (d.confidence * 100).toFixed(1);
    const cls = conf >= 70 ? 'high' : conf >= 50 ? 'mid' : 'low';
    const zone = d.named_zones && d.named_zones.length > 0
      ? '<span class="zone">' + d.named_zones.join(', ') + '</span>' : '';
    const coords = d.coordinates ? `(${d.coordinates[0]},${d.coordinates[1]}) → (${d.coordinates[2]},${d.coordinates[3]})` : '';
    const tid = d.track_id ? ` <span style="color:#58a6ff">ID:${d.track_id}</span>` : '';
    html += `<div class="det-card">
      <div class="time">${d.timestamp || ''}${tid}</div>
      <div class="info">
        <span class="conf ${cls}">${conf}% 人员</span>
        ${zone}
      </div>
      <div class="coords">${coords}</div>
    </div>`;
  });
  list.innerHTML = html || '<div class="empty-state"><div class="icon">🔍</div><div>暂无检测记录</div></div>';
}

// Update stats
function updateStats() {
  document.getElementById('stat-total').textContent = detCount;
  document.getElementById('stat-fps').textContent = document.getElementById('fps-val').textContent;
  
  // Average confidence
  if (allDetections.length > 0) {
    const avg = allDetections.reduce((s,d) => s + d.confidence, 0) / allDetections.length;
    document.getElementById('stat-conf').textContent = (avg * 100).toFixed(1) + '%';
  }
  
  // Recent 5 min
  const fiveMinAgo = Date.now() - 5*60*1000;
  const recent = allDetections.filter(d => {
    try { return new Date(d.timestamp).getTime() > fiveMinAgo; } catch(e) { return false; }
  }).length;
  document.getElementById('stat-recent').textContent = recent;
}

// FPS from MJPEG timing
let lastFrameTime = Date.now();
const videoImg = document.getElementById('video');
videoImg.onload = function() {
  const now = Date.now();
  const delta = now - lastFrameTime;
  if (delta > 0) {
    const fps = (1000 / delta).toFixed(1);
    document.getElementById('fps-val').textContent = fps;
  }
  lastFrameTime = now;
};

// Fetch settings
async function fetchSettings() {
  try {
    const resp = await fetch('/api/dashboard_settings');
    const data = await resp.json();
    let html = '';
    for (const [key, val] of Object.entries(data)) {
      html += `<div class="setting-row"><span class="name">${key}</span><span class="val">${val}</span></div>`;
    }
    document.getElementById('settings-list').innerHTML = html || '<div class="empty-state"><div class="icon">⚙️</div><div>暂无设置信息</div></div>';
  } catch(e) {}
}

// Zone editor
let zoneMode = 'polygon';
let currentPoints = [];
let zoneSnapshot = null;
let existingZones = {polygons: '', lines: ''};

function initZoneEditor() {
  const canvas = document.getElementById('zone-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Load snapshot from video feed
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = '/video_feed';
  img.onload = function() {};

  // Use a static snapshot instead
  canvas.width = 480;
  canvas.height = 360;
  drawZoneCanvas();

  canvas.addEventListener('click', function(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) / rect.width * 100);
    const y = Math.round((e.clientY - rect.top) / rect.height * 100);
    currentPoints.push({x, y});
    drawZoneCanvas();
  });
}

function drawZoneCanvas() {
  const canvas = document.getElementById('zone-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;

  // Dark background
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, w, h);

  // Draw existing zones
  drawConfigZones(ctx, w, h);

  // Draw current points
  ctx.strokeStyle = zoneMode === 'polygon' ? '#3fb950' : '#58a6ff';
  ctx.lineWidth = 2;
  if (currentPoints.length > 0) {
    ctx.beginPath();
    currentPoints.forEach((p, i) => {
      const px = p.x * w / 100;
      const py = p.y * h / 100;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
      // Draw point
      ctx.fillStyle = zoneMode === 'polygon' ? '#3fb950' : '#58a6ff';
      ctx.fillRect(px - 4, py - 4, 8, 8);
      ctx.fillStyle = '#fff';
      ctx.font = '10px monospace';
      ctx.fillText(`${p.x},${p.y}%`, px + 6, py - 4);
    });
    if (zoneMode === 'polygon' && currentPoints.length > 2) {
      ctx.closePath();
      ctx.fillStyle = 'rgba(63, 185, 80, 0.15)';
      ctx.fill();
    }
    ctx.stroke();
  }

  // Update display
  const el = document.getElementById('zone-current');
  if (el) {
    const name = document.getElementById('zone-name')?.value || 'unnamed';
    if (zoneMode === 'polygon') {
      el.textContent = `Polygon "${name}": ${currentPoints.map(p => `${p.x},${p.y}`).join(';')}`;
    } else {
      el.textContent = `Line "${name}": needs 2 points (have ${currentPoints.length})`;
    }
  }
}

function drawConfigZones(ctx, w, h) {
  // Parse and draw existing polygon zones
  if (existingZones.polygons) {
    existingZones.polygons.split('|').forEach(zoneStr => {
      const eqIdx = zoneStr.indexOf('=');
      if (eqIdx < 0) return;
      const coords = zoneStr.substring(eqIdx + 1).trim();
      const points = coords.split(';').map(p => {
        const [x, y] = p.trim().split(',').map(Number);
        return {x, y};
      });
      if (points.length < 3) return;
      ctx.beginPath();
      points.forEach((p, i) => {
        const px = p.x * w / 100, py = p.y * h / 100;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(255, 100, 100, 0.1)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255, 100, 100, 0.4)';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }

  // Parse and draw existing lines
  if (existingZones.lines) {
    existingZones.lines.split('|').forEach(lineStr => {
      const eqIdx = lineStr.indexOf('=');
      if (eqIdx < 0) return;
      const coords = lineStr.substring(eqIdx + 1).trim();
      const parts = coords.split('->');
      if (parts.length !== 2) return;
      const [x1, y1] = parts[0].trim().split(',').map(Number);
      const [x2, y2] = parts[1].trim().split(',').map(Number);
      ctx.beginPath();
      ctx.moveTo(x1 * w / 100, y1 * h / 100);
      ctx.lineTo(x2 * w / 100, y2 * h / 100);
      ctx.strokeStyle = 'rgba(88, 166, 255, 0.4)';
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }
}

function undoPoint() {
  currentPoints.pop();
  drawZoneCanvas();
}

function clearPoints() {
  currentPoints = [];
  drawZoneCanvas();
}

async function saveZones() {
  const name = document.getElementById('zone-name')?.value || 'unnamed';
  if (currentPoints.length < 2) { alert('Need at least 2 points'); return; }

  let newPolygons = existingZones.polygons || '';
  let newLines = existingZones.lines || '';

  if (zoneMode === 'polygon') {
    if (currentPoints.length < 3) { alert('Polygon needs at least 3 points'); return; }
    const coords = currentPoints.map(p => `${p.x},${p.y}`).join(';');
    newPolygons = newPolygons ? `${newPolygons} | ${name}=${coords}` : `${name}=${coords}`;
  } else {
    if (currentPoints.length !== 2) { alert('Line needs exactly 2 points'); return; }
    const coords = `${currentPoints[0].x},${currentPoints[0].y} -> ${currentPoints[1].x},${currentPoints[1].y}`;
    newLines = newLines ? `${newLines} | ${name}=${coords}` : `${name}=${coords}`;
  }

  try {
    const resp = await fetch('/api/zones', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({polygons: newPolygons, lines: newLines})
    });
    const result = await resp.json();
    alert('Saved! ' + result.note);
    existingZones.polygons = newPolygons;
    existingZones.lines = newLines;
    currentPoints = [];
    drawZoneCanvas();
  } catch(e) {
    alert('Save failed: ' + e);
  }
}

async function loadZoneConfig() {
  try {
    const resp = await fetch('/api/zones');
    existingZones = await resp.json();
    drawZoneCanvas();
  } catch(e) {}
}

// Init zone editor when tab is first clicked
let zoneEditorInited = false;
const origSwitchTab = switchTab;
switchTab = function(name) {
  origSwitchTab(name);
  if (name === 'zones' && !zoneEditorInited) {
    zoneEditorInited = true;
    setTimeout(() => { initZoneEditor(); loadZoneConfig(); }, 100);
  }
};
async function fetchPersons() {
  try {
    const resp = await fetch('/api/persons');
    const data = await resp.json();
    if (data.persons && data.persons.length > 0) {
      renderPersons(data.persons);
    } else {
      document.getElementById('persons-list').innerHTML = '<div class="empty-state"><div class="icon">👤</div><div>暂无人员档案</div></div>';
    }
  } catch(e) {}
}

function renderPersons(persons) {
  const list = document.getElementById('persons-list');
  let html = '';
  persons.forEach(p => {
    html += `<div class="det-card" style="display:flex;gap:12px;align-items:center;">
      <img id="photo-${p.id}" style="width:64px;height:64px;border-radius:50%;object-fit:cover;background:#30363d;" src="" alt="">
      <div style="flex:1;">
        <div style="font-size:14px;font-weight:bold;color:#58a6ff;">${p.name}</div>
        <div style="font-size:12px;color:#8b949e;">出现 ${p.count} 次</div>
        <div style="font-size:11px;color:#484f58;">首次: ${p.first_seen}</div>
        <div style="font-size:11px;color:#484f58;">最近: ${p.last_seen}</div>
      </div>
    </div>`;
    // Load photo asynchronously
    fetch(`/api/person_photo/${p.id}`).then(r=>r.json()).then(d=>{
      if(d.photo) document.getElementById(`photo-${p.id}`).src = d.photo;
    }).catch(()=>{});
  });
  list.innerHTML = html;
}

setInterval(fetchDetections, 2000);
setInterval(updateStats, 3000);
setInterval(fetchSettings, 10000);
setInterval(fetchPersons, 5000);
fetchDetections();
fetchSettings();
fetchPersons();
</script>
</body>
</html>
"""

# ============ Flask Routes ============

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/video_feed')
def video_feed():
    """MJPEG stream with aggressive compression for low latency."""
    def generate():
        while True:
            with _frame_lock:
                if _output_frame is None:
                    time.sleep(0.03)
                    continue
                frame = _output_frame.copy()

            # Compress aggressively: resize + low quality JPEG
            h, w = frame.shape[:2]
            # Scale down to max 640px wide for fast transfer
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)  # ~30fps cap for MJPEG

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/detections')
def get_detections():
    with _detections_lock:
        dets = list(_detections_list)
    return jsonify({"detections": dets})


@app.route('/api/persons')
def get_persons():
    """Return all known persons with their photo and stats."""
    import person_db as pdb
    persons = pdb.get_all_persons()
    result = []
    for pid, info in persons.items():
        result.append({
            'id': pid,
            'name': info.get('name', f'人员 #{pid}'),
            'count': info.get('count', 0),
            'first_seen': info.get('first_seen', ''),
            'last_seen': info.get('last_seen', ''),
        })
    return jsonify({"persons": sorted(result, key=lambda x: int(x['id']))})


@app.route('/api/person_photo/<person_id>')
def get_person_photo(person_id):
    """Return person photo as base64."""
    import person_db as pdb
    photo_data = pdb.get_photo_base64(person_id)
    if photo_data:
        return jsonify({"photo": photo_data})
    return jsonify({"photo": None})


@app.route('/api/zones', methods=['GET'])
def get_zones():
    """Return current zone configuration."""
    import configparser
    config = configparser.ConfigParser(interpolation=None)
    config.read('config.ini', encoding='utf-8')
    return jsonify({
        'polygons': config.get('zones', 'polygons', fallback=''),
        'lines': config.get('zones', 'lines', fallback=''),
        'enabled': config.getboolean('zones', 'enabled', fallback=True),
    })


@app.route('/api/zones', methods=['POST'])
def save_zones():
    """Save zone configuration."""
    import configparser
    data = request.get_json()
    config = configparser.ConfigParser(interpolation=None)
    config.read('config.ini', encoding='utf-8')

    if 'polygons' in data:
        config.set('zones', 'polygons', data['polygons'])
    if 'lines' in data:
        config.set('zones', 'lines', data['lines'])

    with open('config.ini', 'w', encoding='utf-8') as f:
        config.write(f)

    return jsonify({"status": "saved", "note": "Restart to apply changes"})


@app.route('/api/dashboard_settings')
def dashboard_settings():
    return jsonify(_config)


# ============ Public API for main script ============

def set_frame(frame):
    global _output_frame
    with _frame_lock:
        _output_frame = frame


def add_detection(detection_info):
    global _detections_list
    with _detections_lock:
        # Convert numpy types to native Python types for JSON serialization
        safe = {}
        for k, v in detection_info.items():
            if hasattr(v, 'item'):  # numpy scalar
                safe[k] = v.item()
            elif isinstance(v, (list, tuple)):
                safe[k] = [x.item() if hasattr(x, 'item') else x for x in v]
            else:
                safe[k] = v
        _detections_list.insert(0, safe)
        _detections_list = _detections_list[:200]  # Keep last 200


def update_config_display(config_dict):
    global _config
    _config = config_dict


def update_fps(fps):
    global _fps
    _fps = fps


def start_dashboard(host='127.0.0.1', port=5001):
    """Start the Chinese dashboard on a separate port."""
    def run():
        serve(app, host=host, port=port, _quiet=True)
    
    t = threading.Thread(target=run, daemon=True)
    t.start()
    url = f"http://{host}:{port}"
    print(f"[监控面板] 已启动: {url}")
    webbrowser.open(url)
