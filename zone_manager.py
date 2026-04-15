#!/usr/bin/env python3
"""
Zone intrusion detection module (Supervision-based)
Zones are configured in config.ini [zones] section with percentage coordinates.

Detection logic: person's BBOX overlaps with the zone polygon (IoU-based).
Alert fires only after a person stays in the zone for `dwell_seconds` (default 3s).
"""

import numpy as np
import cv2
import supervision as sv
from collections import defaultdict
from time import time as _time


def _pct_to_px(shape, pct_points):
    h, w = shape[:2]
    return np.array([[int(x * w / 100), int(y * h / 100)] for x, y in pct_points], dtype=np.int32)


def _box_polygon_overlap(box, polygon):
    """Calculate overlap ratio of a bbox with a polygon (intersection / box_area).
    box: [x1, y1, x2, y2]
    polygon: np.array of shape (N, 2)
    Returns 0.0 ~ 1.0
    """
    x1, y1, x2, y2 = box
    box_area = max((x2 - x1), 1) * max((y2 - y1), 1)
    if box_area == 0:
        return 0.0

    # Create a mask for the polygon
    h = max(int(y2), 1) + 1
    w = max(int(x2), 1) + 1
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_poly = polygon.copy()
    shifted_poly[:, 0] = np.clip(shifted_poly[:, 0], 0, w - 1)
    shifted_poly[:, 1] = np.clip(shifted_poly[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [shifted_poly.astype(np.int32)], 1)

    # Box region
    bx1, by1 = max(int(x1), 0), max(int(y1), 0)
    bx2, by2 = min(int(x2), w), min(int(y2), h)

    if bx2 <= bx1 or by2 <= by1:
        return 0.0

    box_mask = np.zeros_like(mask)
    box_mask[by1:by2, bx1:bx2] = 1

    intersection = np.logical_and(mask, box_mask).sum()
    return float(intersection) / float(box_area)


class ZoneManager:
    def __init__(self, frame_shape, config=None):
        self._frame_shape = frame_shape
        h, w = frame_shape[:2]
        self.zones = {}
        self.zone_annotators = {}
        self.zone_counts = defaultdict(int)
        self.zone_current = {}

        # Dwell tracking: {zone_name: {track_id: first_seen_timestamp}}
        self.dwell_start = defaultdict(dict)
        # How many seconds a person must stay in zone before alert
        self.dwell_seconds = 3.0
        # Overlap threshold: bbox must overlap this much with zone (0.0 ~ 1.0)
        self.overlap_threshold = 0.3
        # FPS estimate for frame-based fallback
        self._fps = 20.0
        # Already-alerted track+zone pairs (avoid repeated alerts)
        self._alerted = set()

        # Keep sv PolygonZone for annotation only
        self.sv_zones = {}

        default_zones = {}  # No default zones - empty means no zones

        zone_configs = default_zones
        if config:
            try:
                raw = config.get('zones', 'polygons', fallback='')
                if raw:
                    zone_configs = {}
                    for item in raw.split('|'):
                        if '=' in item:
                            n, c = item.split('=', 1)
                            zone_configs[n.strip()] = c.strip()
                # Read dwell_seconds from config
                ds = config.getfloat('zones', 'dwell_seconds', fallback=3.0)
                if ds > 0:
                    self.dwell_seconds = ds
                ot = config.getfloat('zones', 'overlap_threshold', fallback=0.3)
                if 0 < ot <= 1:
                    self.overlap_threshold = ot
            except Exception:
                pass

        for name, coords_str in zone_configs.items():
            try:
                points = []
                for pair in coords_str.split(';'):
                    x, y = pair.strip().split(',')
                    points.append([float(x), float(y)])
                polygon = _pct_to_px(frame_shape, points)

                self.zones[name] = polygon
                self.sv_zones[name] = sv.PolygonZone(
                    polygon=polygon,
                    triggering_anchors=[sv.Position.CENTER],
                )
                self.zone_annotators[name] = sv.PolygonZoneAnnotator(
                    zone=self.sv_zones[name],
                    color=sv.Color.RED,
                    thickness=2,
                    text_thickness=1,
                    text_scale=0.5,
                )
                self.zone_current[name] = 0
            except Exception as e:
                print(f"[Zone] Failed to load '{name}': {e}")

        print(f"[Zone] Loaded {len(self.zones)} zones (dwell={self.dwell_seconds}s, overlap={self.overlap_threshold})")

    def reload(self, config=None):
        """Hot-reload zones from config without restarting."""
        self.zones.clear()
        self.sv_zones.clear()
        self.zone_annotators.clear()
        self.zone_counts.clear()
        self.zone_current.clear()
        self.dwell_start.clear()
        self._alerted.clear()

        zone_configs = {}
        if config:
            try:
                raw = config.get('zones', 'polygons', fallback='')
                if raw:
                    for item in raw.split('|'):
                        if '=' in item:
                            n, c = item.split('=', 1)
                            zone_configs[n.strip()] = c.strip()
            except Exception:
                pass

        for name, coords_str in zone_configs.items():
            try:
                points = []
                for pair in coords_str.split(';'):
                    x, y = pair.strip().split(',')
                    points.append([float(x), float(y)])
                polygon = _pct_to_px(self._frame_shape, points)

                self.zones[name] = polygon
                self.sv_zones[name] = sv.PolygonZone(
                    polygon=polygon,
                    triggering_anchors=[sv.Position.CENTER],
                )
                self.zone_annotators[name] = sv.PolygonZoneAnnotator(
                    zone=self.sv_zones[name],
                    color=sv.Color.RED,
                    thickness=2,
                    text_thickness=1,
                    text_scale=0.5,
                )
                self.zone_current[name] = 0
            except Exception as e:
                print(f"[Zone] Failed to load '{name}': {e}")

        print(f"[Zone] Reloaded {len(self.zones)} zones")

    def _bbox_in_zone(self, bbox, polygon):
        """Check if bbox overlaps with zone polygon above threshold."""
        overlap = _box_polygon_overlap(bbox, polygon)
        return overlap >= self.overlap_threshold

    def update(self, detections, fps=None):
        """Update zone detection. Returns zone_alerts for dwell-triggered intrusions."""
        if fps:
            self._fps = fps

        now = _time()
        zone_alerts = []
        prev_counts = dict(self.zone_current)

        # Get tracker_ids if available
        tracker_ids = detections.tracker_id if hasattr(detections, 'tracker_id') and detections.tracker_id is not None else None

        for name, polygon in self.zones.items():
            current_in_zone = 0
            # Track which track_ids are currently in this zone
            active_tracks = set()

            for i in range(len(detections)):
                # Get bbox
                if hasattr(detections, 'xyxy') and detections.xyxy is not None and len(detections.xyxy) > i:
                    bbox = detections.xyxy[i]
                else:
                    continue

                if self._bbox_in_zone(bbox, polygon):
                    current_in_zone += 1
                    tid = int(tracker_ids[i]) if tracker_ids is not None and i < len(tracker_ids) else i
                    active_tracks.add(tid)

                    # Track dwell time
                    if tid not in self.dwell_start[name]:
                        self.dwell_start[name][tid] = now

                    dwell = now - self.dwell_start[name][tid]
                    # Fire alert if dwell exceeded and not already alerted
                    if dwell >= self.dwell_seconds and (name, tid) not in self._alerted:
                        self._alerted.add((name, tid))
                        zone_alerts.append((name, current_in_zone))

            # Clean up tracks that left the zone
            left_tracks = set(self.dwell_start[name].keys()) - active_tracks
            for tid in left_tracks:
                del self.dwell_start[name][tid]
                self._alerted.discard((name, tid))

            self.zone_current[name] = current_in_zone
            if current_in_zone > 0:
                self.zone_counts[name] += current_in_zone

            # Sync sv.PolygonZone internal counter so annotator shows correct number
            if name in self.sv_zones:
                self.sv_zones[name].current_count = current_in_zone

        return {'zone_alerts': zone_alerts}

    def annotate(self, frame, detections):
        for name in self.sv_zones:
            frame = self.zone_annotators[name].annotate(scene=frame)

        y = 60
        for name, count in self.zone_current.items():
            total = self.zone_counts[name]
            # Show dwell info
            dwell_info = ""
            if name in self.dwell_start:
                now = _time()
                dwell_times = [now - t for t in self.dwell_start[name].values()]
                if dwell_times:
                    max_dwell = max(dwell_times)
                    dwell_info = f" max:{max_dwell:.1f}s"
            cv2.putText(frame,
                f"#{list(self.zones.keys()).index(name)+1} {name}: {count} in ({total} total){dwell_info}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
            y += 25

        return frame

    def get_stats(self):
        stats = {}
        for name in self.zones:
            stats[name] = {
                'current': self.zone_current.get(name, 0),
                'total': self.zone_counts.get(name, 0),
            }
        return {'zones': stats, 'lines': {}}
