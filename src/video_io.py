from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoSource:
    path: Optional[str] = None
    camera_index: Optional[int] = None


def open_capture(source: VideoSource) -> cv2.VideoCapture:
    if source.path:
        return cv2.VideoCapture(source.path)
    if source.camera_index is not None:
        return cv2.VideoCapture(source.camera_index)
    raise ValueError("Provide either a video path or camera index.")


def iter_sampled_frames(
    cap: cv2.VideoCapture, sample_fps: int
) -> Generator[Tuple[int, np.ndarray], None, None]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = sample_fps

    frame_interval = max(int(round(fps / sample_fps)), 1)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            yield frame_idx, frame
        frame_idx += 1
