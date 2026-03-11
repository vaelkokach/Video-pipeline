from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    bbox: List[float]
    score: float
    label: int


def _build_yolo_counter(device: str) -> "YOLOCounter":
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Person detection requires ultralytics (YOLO). Run: pip install ultralytics"
        ) from exc
    model = YOLO("yolov8n.pt")  # Nano model, auto-downloads, COCO includes person (class 0)
    return YOLOCounter(model=model, device=device)


def _build_mmdet_counter(
    det_config: str, det_checkpoint: Optional[str], device: str
) -> "MMDetCounter":
    from mmdet.apis import DetInferencer

    kwargs = {"model": str(det_config), "device": device}
    if det_checkpoint:
        kwargs["weights"] = str(det_checkpoint)
    inferencer = DetInferencer(**kwargs)
    return MMDetCounter(inferencer=inferencer)


class YOLOCounter:
    """Person detection using YOLOv8 (works without mmcv full)."""

    PERSON_CLASS = 0  # COCO person

    def __init__(self, model, device: str) -> None:
        self.model = model
        self.device = device

    def count_people(self, frame: np.ndarray, score_thr: float) -> int:
        return len(self.detect_people(frame, score_thr))

    def detect_people(self, frame: np.ndarray, score_thr: float) -> List[List[float]]:
        results = self.model(frame, verbose=False, device=self.device)
        out: List[List[float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                cls = int(r.boxes.cls[i].item())
                if cls != self.PERSON_CLASS:
                    continue
                conf = float(r.boxes.conf[i].item())
                if conf < score_thr:
                    continue
                xyxy = r.boxes.xyxy[i].cpu().numpy()
                out.append(xyxy.tolist())
        return out


class MMDetCounter:
    """Person detection using MMDetection (requires full mmcv with CUDA ops)."""

    def __init__(self, inferencer) -> None:
        self._inferencer = inferencer

    def count_people(self, frame: np.ndarray, score_thr: float) -> int:
        return len(self.detect_people(frame, score_thr))

    def detect_people(self, frame: np.ndarray, score_thr: float) -> List[List[float]]:
        result = self._inferencer(frame, return_datasamples=True)
        preds = result["predictions"][0]
        bboxes = preds.bboxes
        labels = preds.labels
        scores = preds.scores
        is_person = labels == 0
        over_thr = scores >= score_thr
        keep = is_person & over_thr
        kept_bboxes = bboxes[keep]
        return [box.tolist() for box in kept_bboxes]


class StudentCounter:
    def __init__(
        self,
        det_config: str,
        det_checkpoint: Optional[str],
        device: str,
    ) -> None:
        self.det_config = det_config
        self.det_checkpoint = det_checkpoint
        self.device = device
        self._counter = self._build_counter()

    def _build_counter(self):
        # Prefer YOLO for compatibility (no mmcv compiled ops). Use MMDet when explicitly configured.
        use_yolo = self.det_config.lower() in ("yolo", "yolov8", "yolov8n")
        if use_yolo:
            return _build_yolo_counter(self.device)
        try:
            return _build_mmdet_counter(
                self.det_config, self.det_checkpoint, self.device
            )
        except Exception:
            # Fallback to YOLO if MMDet fails (e.g. mmcv-lite)
            return _build_yolo_counter(self.device)

    def count_people(self, frame: np.ndarray, score_thr: float) -> int:
        return self._counter.count_people(frame, score_thr)

    def detect_people(self, frame: np.ndarray, score_thr: float) -> List[List[float]]:
        return self._counter.detect_people(frame, score_thr)
