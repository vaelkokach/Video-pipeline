from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class EmotionPrediction:
    label: str
    score: float


class EmotionRecognizer:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name or "enet_b0_8_best_vgaf"
        self.device = device
        self._model = self._build_model()

    def _build_model(self):
        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizerTorch
        except ImportError as exc:
            raise RuntimeError(
                "EmotiEffLib is required for emotion recognition. "
                "Install emotiefflib and update requirements."
            ) from exc
        model_name = self.model_name
        if model_name in ("default", "emotiefflib_default"):
            model_name = "enet_b0_8_best_vgaf"
        return EmotiEffLibRecognizerTorch(model_name=model_name, device=self.device)

    def predict_frame(self, frame: np.ndarray) -> List[EmotionPrediction]:
        # EmotiEffLib expects BGR/RGB image; predict_emotions returns (labels, scores)
        labels, scores = self._model.predict_emotions(frame, logits=False)
        predictions: List[EmotionPrediction] = []
        if isinstance(labels, str):
            labels = [labels]
            scores = scores[np.newaxis, ...]
        for i, label in enumerate(labels):
            score = float(scores[i].max()) if scores.size > 0 else 0.0
            predictions.append(EmotionPrediction(label=str(label), score=score))
        return predictions
