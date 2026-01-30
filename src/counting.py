from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Detection:
    bbox: List[float]
    score: float
    label: int


class StudentCounter:
    def __init__(self, det_config: str, det_checkpoint: str, device: str) -> None:
        self.det_config = det_config
        self.det_checkpoint = det_checkpoint
        self.device = device
        self._inferencer = self._build_inferencer()

    def _build_inferencer(self):
        try:
            from mmdet.apis import DetInferencer
        except ImportError as exc:
            raise RuntimeError(
                "MMDetection is required for counting. Install mmdet and "
                "update requirements."
            ) from exc
        return DetInferencer(
            model=str(self.det_config),
            weights=str(self.det_checkpoint),
            device=self.device,
        )

    def count_people(self, frame: np.ndarray, score_thr: float) -> int:
        detections = self.detect_people(frame, score_thr)
        return len(detections)

    def detect_people(self, frame: np.ndarray, score_thr: float) -> List[List[float]]:
        result = self._inferencer(frame, return_datasamples=True)
        preds = result["predictions"][0]

        bboxes = preds.bboxes
        labels = preds.labels
        scores = preds.scores

        # COCO person class id is 0 for standard models.
        is_person = labels == 0
        over_thr = scores >= score_thr

        keep = is_person & over_thr
        kept_bboxes = bboxes[keep]
        return [box.tolist() for box in kept_bboxes]
