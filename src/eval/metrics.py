from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..attention.calibration import expected_calibration_error


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_class: List[Dict[str, float]] = []
    f1_scores: List[float] = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class.append({"precision": precision, "recall": recall, "f1": f1})
        f1_scores.append(f1)

    accuracy = _safe_div(float(np.trace(cm)), float(cm.sum()))
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class, "confusion_matrix": cm.tolist()}


def _binary_curve_scores(y_true_bin: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y = y_true_bin[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    tpr = tp / positives
    fpr = fp / negatives
    return fpr, tpr


def auroc_ovr(y_true: np.ndarray, probs: np.ndarray, positive_class: int = 3) -> float:
    y_bin = (y_true == positive_class).astype(np.int64)
    score = probs[:, positive_class]
    fpr, tpr = _binary_curve_scores(y_bin, score)
    return float(np.trapz(tpr, fpr))


def auprc_ovr(y_true: np.ndarray, probs: np.ndarray, positive_class: int = 3) -> float:
    y_bin = (y_true == positive_class).astype(np.int64)
    score = probs[:, positive_class]
    order = np.argsort(-score)
    y = y_bin[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.clip(tp + fp, 1, None)
    recall = tp / max(int((y == 1).sum()), 1)
    return float(np.trapz(precision, recall))


def ece_from_probs(y_true: np.ndarray, probs: np.ndarray) -> float:
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correctness = (pred == y_true).astype(np.float32)
    return expected_calibration_error(confidence=confidence, correctness=correctness, bins=10)


def flip_rate(y_pred_seq: np.ndarray) -> float:
    if len(y_pred_seq) < 2:
        return 0.0
    return float(np.mean(y_pred_seq[1:] != y_pred_seq[:-1]))


def time_to_detect(y_true_seq: np.ndarray, y_pred_seq: np.ndarray, positive_class: int = 3) -> float:
    true_idx = np.where(y_true_seq == positive_class)[0]
    pred_idx = np.where(y_pred_seq == positive_class)[0]
    if len(true_idx) == 0 or len(pred_idx) == 0:
        return float("inf")
    return float(max(0, pred_idx[0] - true_idx[0]))
