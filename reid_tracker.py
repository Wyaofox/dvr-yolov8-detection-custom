#!/usr/bin/env python3
"""
轻量行人重识别模块
- 使用 OSNet 提取行人外观特征
- 跨帧匹配：人离开画面再回来，仍能识别为同一人
- 与 ByteTrack 的 track_id 做关联映射
"""

import numpy as np
import cv2
import threading
import time

# Lazy-load torchreid to avoid slow startup when not needed
_reid_model = None
_reid_lock = threading.Lock()


def _get_model():
    """Lazy-load the OSNet ReID model."""
    global _reid_model
    if _reid_model is not None:
        return _reid_model

    try:
        import torch
        from torchreid.utils import FeatureExtractor

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        extractor = FeatureExtractor(
            model_name='osnet_x0_25',  # Lightweight OSNet variant
            model_path='',
            device=device,
        )
        _reid_model = extractor
        print("[ReID] OSNet x0.25 模型已加载")
        return _reid_model
    except Exception as e:
        print(f"[ReID] 模型加载失败: {e}")
        _reid_model = False  # Mark as unavailable
        return None


def extract_feature(frame, bbox):
    """
    Extract appearance feature from a person crop.
    bbox: (x1, y1, x2, y2)
    Returns: feature vector (numpy array) or None
    """
    model = _get_model()
    if model is None or model is False:
        return None

    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Clamp to frame bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None  # Too small

    crop = frame[y1:y2, x1:x2]
    try:
        # Resize to standard ReID input size
        crop_resized = cv2.resize(crop, (128, 256))
        # torchreid expects BGR -> RGB
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        features = model(crop_rgb)
        feat = features.cpu().numpy().flatten()
        # Normalize
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat
    except Exception as e:
        return None


def cosine_similarity(a, b):
    """Compute cosine similarity between two feature vectors."""
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


class ReIDTracker:
    """
    Maintains a gallery of known person features.
    Maps ByteTrack IDs to persistent ReID identities.
    """

    def __init__(self, similarity_threshold=0.45, max_gallery_size=50, ttl_seconds=600):
        """
        similarity_threshold: cosine similarity threshold for matching (0-1)
        max_gallery_size: max number of known persons to keep
        ttl_seconds: how long to remember a person after they leave (10 min)
        """
        self.threshold = similarity_threshold
        self.max_gallery = max_gallery_size
        self.ttl = ttl_seconds

        # reid_id -> {'feature': np.array, 'last_seen': float, 'track_ids': set}
        self.gallery = {}
        self._next_reid_id = 1
        self._lock = threading.Lock()

        # Preload model in background
        t = threading.Thread(target=_get_model, daemon=True)
        t.start()

    def _clean_expired(self):
        """Remove gallery entries older than TTL."""
        now = time.time()
        expired = [k for k, v in self.gallery.items() if now - v['last_seen'] > self.ttl]
        for k in expired:
            print(f"[ReID] 身份 #{k} 已过期（{self.ttl}秒未出现）")
            del self.gallery[k]

    def match(self, feature):
        """
        Match a feature against the gallery.
        Returns: reid_id if matched, None if no match
        """
        if feature is None:
            return None

        best_id = None
        best_sim = 0.0

        for reid_id, entry in self.gallery.items():
            sim = cosine_similarity(feature, entry['feature'])
            if sim > best_sim:
                best_sim = sim
                best_id = reid_id

        if best_sim >= self.threshold:
            return best_id
        return None

    def update(self, track_id, frame, bbox):
        """
        Update the tracker with a new detection.
        Returns: (reid_id, is_new_person)
        """
        model = _get_model()
        if model is None or model is False:
            # ReID unavailable, fall back to track_id
            return track_id, True

        feature = extract_feature(frame, bbox)
        if feature is None:
            return track_id, True

        with self._lock:
            self._clean_expired()

            # Try to match existing identity
            matched_reid = self.match(feature)

            if matched_reid is not None:
                # Found a match! Update gallery
                entry = self.gallery[matched_reid]
                # Update feature with exponential moving average
                entry['feature'] = 0.7 * entry['feature'] + 0.3 * feature
                norm = np.linalg.norm(entry['feature'])
                if norm > 0:
                    entry['feature'] = entry['feature'] / norm
                entry['last_seen'] = time.time()
                entry['track_ids'].add(track_id)

                is_returning = track_id not in entry.get('known_track_ids', {track_id})
                if not hasattr(entry, 'known_track_ids'):
                    entry['known_track_ids'] = {track_id}
                else:
                    entry['known_track_ids'].add(track_id)

                return matched_reid, False
            else:
                # New person
                reid_id = self._next_reid_id
                self._next_reid_id += 1

                self.gallery[reid_id] = {
                    'feature': feature,
                    'last_seen': time.time(),
                    'track_ids': {track_id},
                    'known_track_ids': {track_id},
                }

                # Trim gallery if too large
                if len(self.gallery) > self.max_gallery:
                    oldest = min(self.gallery.items(), key=lambda x: x[1]['last_seen'])
                    del self.gallery[oldest[0]]

                return reid_id, True

    def get_known_count(self):
        """Return the number of known unique persons."""
        with self._lock:
            return len(self.gallery)


# Global instance
reid_tracker = ReIDTracker()
