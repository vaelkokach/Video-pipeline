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
        self.model_name = model_name
        self.device = device
        self._model = self._build_model()

    def _build_model(self):
        try:
            from emotiefflib import EmotiEffLib
        except ImportError as exc:
            raise RuntimeError(
                "EmotiEffLib is required for emotion recognition. "
                "Install emotiefflib and update requirements."
            ) from exc
        return EmotiEffLib(model_name=self.model_name, device=self.device)

    def predict_frame(self, frame: np.ndarray) -> List[EmotionPrediction]:
        # EmotiEffLib typically returns a list of predictions per detected face.
        result = self._model.predict(frame)
        predictions: List[EmotionPrediction] = []

        for face_pred in result:
            label = face_pred.get("label")
            score = face_pred.get("score", 0.0)
            predictions.append(EmotionPrediction(label=str(label), score=float(score)))

        return predictions
