import numpy as np

from src.eval.significance import bootstrap_ci, paired_permutation_test


def test_bootstrap_ci_monotonic():
    values = np.array([0, 1, 1, 0, 1, 1, 0, 1], dtype=np.float32)
    lo, hi = bootstrap_ci(values, metric_fn=lambda x: float(np.mean(x)), n_bootstrap=300, seed=7)
    assert 0.0 <= lo <= hi <= 1.0


def test_paired_permutation_reasonable():
    a = np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=np.float32)
    b = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    p_value = paired_permutation_test(a, b, n_permutations=1000, seed=7)
    assert 0.0 <= p_value <= 1.0
