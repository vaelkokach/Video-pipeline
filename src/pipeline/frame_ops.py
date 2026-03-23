from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np


def crop_and_resize(frame: np.ndarray, bbox: List[float], size: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (size, size))


def maybe_resize_frame(frame: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    if width is None or height is None:
        return frame
    return cv2.resize(frame, (width, height))


def draw_annotations(
    frame: np.ndarray,
    assignments: Dict[int, List[float]],
    current_labels: Dict[int, Dict[str, Optional[str]]],
) -> np.ndarray:
    """Draw bounding boxes and labels (student#, action, emotion) on frame."""
    out = frame.copy()
    for track_id, bbox in assignments.items():
        x1, y1, x2, y2 = [int(v) for v in bbox]
        labels = current_labels.get(track_id, {})
        action = labels.get("action") or "-"
        emotion = labels.get("emotion") or "-"
        text = f"#{track_id} | A:{action} E:{emotion}"
        color = (0, 200, 100)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly1 = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, ly1), (min(x1 + tw + 4, out.shape[1]), y1), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return out
