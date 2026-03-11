from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class ActionPrediction:
    label: str
    score: float


def _clip_to_video(clip: np.ndarray, fps: int = 8) -> Path:
    """Write a clip (T, H, W, C) to a temporary video file for MMAction2 pipeline."""
    t, h, w, c = clip.shape
    path = Path(tempfile.mkdtemp()) / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(t):
        frame = clip[i]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


class ActionRecognizer:
    def __init__(self, config: str, checkpoint: str, device: str) -> None:
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self._model = self._build_model()

    def _build_model(self):
        try:
            from mmaction.apis import init_recognizer
        except ImportError as exc:
            raise RuntimeError(
                "MMACTION2 is required for action recognition. "
                "Install mmaction2 and update requirements."
            ) from exc
        return init_recognizer(str(self.config), str(self.checkpoint), device=self.device)

    def predict_clip(self, clip: np.ndarray) -> List[ActionPrediction]:
        try:
            from mmaction.apis import inference_recognizer
        except ImportError as exc:
            raise RuntimeError(
                "MMACTION2 is required for action recognition. "
                "Install mmaction2 and update requirements."
            ) from exc

        video_path = _clip_to_video(clip)
        try:
            result = inference_recognizer(self._model, str(video_path))
        finally:
            video_path.unlink(missing_ok=True)
            video_path.parent.rmdir(missing_ok=True)
        predictions: List[ActionPrediction] = []

        scores = getattr(result, "pred_score", None)
        if scores is None:
            scores = getattr(result, "pred_scores", None)

        if scores is None:
            raise RuntimeError(
                "Unexpected MMACTION2 output format. Update ActionRecognizer "
                "to match your model output."
            )

        # Handle tensor/array: (num_classes,) -> list of (class_idx, score)
        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
        scores = np.asarray(scores).ravel()
        for idx, score in enumerate(scores):
            predictions.append(ActionPrediction(label=str(idx), score=float(score)))

        predictions.sort(key=lambda p: p.score, reverse=True)
        return predictions
