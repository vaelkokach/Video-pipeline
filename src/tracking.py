from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def _iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class Track:
    track_id: int
    bbox: List[float]
    last_seen: int


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    def update(
        self, detections: List[List[float]], frame_index: int
    ) -> Tuple[Dict[int, List[float]], List[int]]:
        assignments: Dict[int, List[float]] = {}

        track_ids = list(self._tracks.keys())
        used_tracks = set()
        used_dets = set()

        for det_idx, det in enumerate(detections):
            best_iou = 0.0
            best_track_id = None
            for track_id in track_ids:
                if track_id in used_tracks:
                    continue
                iou = _iou(self._tracks[track_id].bbox, det)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                used_tracks.add(best_track_id)
                used_dets.add(det_idx)
                self._tracks[best_track_id].bbox = det
                self._tracks[best_track_id].last_seen = frame_index
                assignments[best_track_id] = det

        for det_idx, det in enumerate(detections):
            if det_idx in used_dets:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = Track(track_id=track_id, bbox=det, last_seen=frame_index)
            assignments[track_id] = det

        expired = self._expire_old_tracks(frame_index)
        return assignments, expired

    def _expire_old_tracks(self, frame_index: int) -> List[int]:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_seen > self.max_age
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return expired