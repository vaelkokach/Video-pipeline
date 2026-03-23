import numpy as np

from src.eval.metrics import (
    auprc_ovr,
    auroc_ovr,
    classification_report,
    ece_from_probs,
    flip_rate,
    time_to_detect,
)


def test_classification_report_shapes():
    y_true = np.array([0, 1, 2, 3, 2, 1, 0, 3])
    y_pred = np.array([0, 1, 1, 3, 2, 1, 0, 2])
    report = classification_report(y_true, y_pred, num_classes=4)
    assert "accuracy" in report
    assert "macro_f1" in report
    assert len(report["per_class"]) == 4


def test_curve_metrics_in_range():
    y_true = np.array([0, 1, 2, 3, 3, 2, 1, 0])
    probs = np.array(
        [
            [0.8, 0.1, 0.1, 0.0],
            [0.2, 0.6, 0.1, 0.1],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.1, 0.1, 0.7],
            [0.1, 0.1, 0.2, 0.6],
            [0.2, 0.1, 0.6, 0.1],
            [0.2, 0.5, 0.2, 0.1],
            [0.7, 0.1, 0.1, 0.1],
        ]
    )
    auroc = auroc_ovr(y_true, probs, positive_class=3)
    auprc = auprc_ovr(y_true, probs, positive_class=3)
    ece = ece_from_probs(y_true, probs)
    assert 0.0 <= auroc <= 1.0
    assert 0.0 <= auprc <= 1.0
    assert 0.0 <= ece <= 1.0


def test_temporal_metrics():
    y_true_seq = np.array([1, 1, 3, 3, 3])
    y_pred_seq = np.array([1, 2, 2, 3, 3])
    assert flip_rate(y_pred_seq) > 0.0
    assert time_to_detect(y_true_seq, y_pred_seq, positive_class=3) == 1.0
