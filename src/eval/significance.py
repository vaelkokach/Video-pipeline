from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = []
    n = len(values)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(values[idx]))
    lo = np.percentile(stats, (alpha / 2) * 100)
    hi = np.percentile(stats, (1 - alpha / 2) * 100)
    return float(lo), float(hi)


def paired_permutation_test(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42,
) -> float:
    """
    Two-sided paired permutation test for improvement significance.
    """
    rng = np.random.default_rng(seed)
    observed = float(np.mean(metric_a - metric_b))
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(metric_a))
        perm = np.mean((metric_a - metric_b) * signs)
        if abs(perm) >= abs(observed):
            count += 1
    return float((count + 1) / (n_permutations + 1))
