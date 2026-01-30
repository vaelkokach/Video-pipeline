from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ActionPrediction:
    label: str
    score: float


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

        result = inference_recognizer(self._model, clip)
        predictions: List[ActionPrediction] = []

        scores = getattr(result, "pred_score", None)
        if scores is None:
            scores = getattr(result, "pred_scores", None)

        if isinstance(scores, dict):
            items = scores.items()
        elif scores is not None and hasattr(result, "pred_label"):
            items = zip(result.pred_label, scores)
        else:
            raise RuntimeError(
                "Unexpected MMACTION2 output format. Update ActionRecognizer "
                "to match your model output."
            )

        for label, score in items:
            predictions.append(ActionPrediction(label=str(label), score=float(score)))

        predictions.sort(key=lambda p: p.score, reverse=True)
        return predictions
