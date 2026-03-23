from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .metrics import classification_report
from .significance import bootstrap_ci, paired_permutation_test


@dataclass
class AblationResult:
    name: str
    macro_f1: float
    accuracy: float


def compare_models(y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, object]:
    rows: List[AblationResult] = []
    for name, y_pred in predictions.items():
        rep = classification_report(y_true, y_pred, num_classes=4)
        rows.append(AblationResult(name=name, macro_f1=float(rep["macro_f1"]), accuracy=float(rep["accuracy"])))

    rows_sorted = sorted(rows, key=lambda r: r.macro_f1, reverse=True)
    best = rows_sorted[0]
    baseline = rows_sorted[-1]

    p_value = paired_permutation_test(
        metric_a=(predictions[best.name] == y_true).astype(np.float32),
        metric_b=(predictions[baseline.name] == y_true).astype(np.float32),
    )
    ci = bootstrap_ci(
        values=(predictions[best.name] == y_true).astype(np.float32),
        metric_fn=lambda x: float(np.mean(x)),
    )
    return {
        "ranking": [r.__dict__ for r in rows_sorted],
        "best_vs_baseline_p_value": p_value,
        "best_accuracy_bootstrap_ci": {"low": ci[0], "high": ci[1]},
    }


def write_ablation_report(output_path: str | Path, payload: Dict[str, object]) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
