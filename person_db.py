#!/usr/bin/env python3
"""
人员数据库模块 - 基于 DeepFace
- 检测到新人时截图保存
- 用 DeepFace (ArcFace) 比对是否为已知人员
- 管理人员档案：照片、出现次数、首次/最近出现时间
- 为 dashboard 提供人员列表 API
"""

import cv2
import os
import json
import time
import threading
import base64
import numpy as np
from datetime import datetime

# Person database directory
PERSON_DB_DIR = None
PERSON_META_FILE = None
_person_db = {}  # person_id -> {photo_path, count, first_seen, last_seen, name}
_db_lock = threading.Lock()
_next_id = 1
_deepface_ready = False


def init_person_db(base_dir):
    """Initialize the person database directory and load existing data."""
    global PERSON_DB_DIR, PERSON_META_FILE, _person_db, _next_id, _deepface_ready

    PERSON_DB_DIR = os.path.join(base_dir, "persons")
    PERSON_META_FILE = os.path.join(PERSON_DB_DIR, "persons.json")
    os.makedirs(PERSON_DB_DIR, exist_ok=True)

    # Load existing database
    if os.path.exists(PERSON_META_FILE):
        try:
            with open(PERSON_META_FILE, 'r', encoding='utf-8') as f:
                _person_db = json.load(f)
            if _person_db:
                _next_id = max(int(k) for k in _person_db.keys()) + 1
                print(f"[人员库] 已加载 {len(_person_db)} 个人员档案")
        except Exception as e:
            print(f"[人员库] 加载失败: {e}")
            _person_db = {}

    # Preload DeepFace in background
    t = threading.Thread(target=_preload_deepface, daemon=True)
    t.start()


def _preload_deepface():
    """Preload DeepFace models in background."""
    global _deepface_ready
    try:
        from deepface import DeepFace
        # Force model download by doing a dummy operation
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        _deepface_ready = True
        print("[DeepFace] 模型就绪 (ArcFace)")
    except Exception as e:
        print(f"[DeepFace] 加载失败: {e}")
        _deepface_ready = False


def _save_db():
    """Save person database to disk."""
    if PERSON_META_FILE:
        try:
            with open(PERSON_META_FILE, 'w', encoding='utf-8') as f:
                json.dump(_person_db, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[人员库] 保存失败: {e}")


def _crop_person(frame, bbox, margin=30):
    """Crop person region from frame with margin."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    return frame[y1:y2, x1:x2]


def _save_person_photo(person_id, crop):
    """Save person photo to disk."""
    if not PERSON_DB_DIR:
        return None
    filename = f"person_{person_id}.jpg"
    filepath = os.path.join(PERSON_DB_DIR, filename)
    cv2.imwrite(filepath, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return filename


def find_match(crop_path, threshold=0.4):
    """
    Use DeepFace to check if this person matches anyone in the database.
    Returns: person_id if match found, None otherwise.
    """
    if not _deepface_ready:
        return None

    try:
        from deepface import DeepFace
    except ImportError:
        return None

    with _db_lock:
        if not _person_db:
            return None
        db_snapshot = dict(_person_db)

    for person_id, info in db_snapshot.items():
        ref_path = os.path.join(PERSON_DB_DIR, info.get('photo_path', ''))
        if not os.path.exists(ref_path):
            continue

        try:
            result = DeepFace.verify(
                img1_path=crop_path,
                img2_path=ref_path,
                model_name="ArcFace",
                distance_metric="cosine",
                enforce_detection=False,  # Don't fail if face not detected
                align=True,
            )
            # Lower distance = more similar; threshold controls match strictness
            if result['distance'] < threshold and result['verified']:
                return person_id
        except Exception:
            continue

    return None


def register_or_match(frame, bbox, track_id=None):
    """
    Main entry: crop person, save photo, match against database.
    Returns: (person_id, is_new_person)
    """
    global _next_id

    crop = _crop_person(frame, bbox)
    if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
        return None, True

    with _db_lock:
        # Save crop to temp file for DeepFace comparison
        temp_path = os.path.join(PERSON_DB_DIR or "/tmp", "_temp_match.jpg")
        cv2.imwrite(temp_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Try to match against existing persons
        matched_id = find_match(temp_path)

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if matched_id is not None:
            # Known person - update record
            info = _person_db[matched_id]
            info['count'] += 1
            info['last_seen'] = now
            info['last_track_id'] = track_id
            _save_db()
            return matched_id, False
        else:
            # New person
            person_id = str(_next_id)
            _next_id += 1

            photo_filename = _save_person_photo(person_id, crop)

            _person_db[person_id] = {
                'photo_path': photo_filename,
                'count': 1,
                'first_seen': now,
                'last_seen': now,
                'last_track_id': track_id,
                'name': f"人员 #{person_id}",
            }
            _save_db()
            print(f"[人员库] 新人员 #{person_id} 已登记")
            return person_id, True


def get_all_persons():
    """Return all persons in the database for dashboard display."""
    with _db_lock:
        return dict(_person_db)


def get_person_count():
    """Return total number of unique persons."""
    with _db_lock:
        return len(_person_db)


def get_photo_base64(person_id):
    """Return base64 encoded photo for dashboard."""
    with _db_lock:
        info = _person_db.get(person_id)
        if not info:
            return None
        photo_path = os.path.join(PERSON_DB_DIR or "", info.get('photo_path', ''))
        if not os.path.exists(photo_path):
            return None
        _, ext = os.path.splitext(photo_path)
        with open(photo_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        return f"data:image/{ext[1:]};base64,{data}"
