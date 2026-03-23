from __future__ import annotations

import numpy as np


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    return logits / temperature


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-9, None)


def expected_calibration_error(confidence: np.ndarray, correctness: np.ndarray, bins: int = 10) -> float:
    ece = 0.0
    n = len(confidence)
    if n == 0:
        return 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < bins - 1 else confidence <= hi)
        if not np.any(mask):
            continue
        acc = correctness[mask].mean()
        conf = confidence[mask].mean()
        ece += (mask.mean()) * abs(float(acc - conf))
    return float(ece)
