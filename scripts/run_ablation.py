from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.eval.ablation import compare_models, write_ablation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation comparison for thesis results.")
    parser.add_argument("--output", type=str, default="outputs/eval/ablation_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    y_true = rng.integers(0, 4, size=args.n)

    # Synthetic benchmark harness (replace with true model predictions in experiments).
    pred_ce = y_true.copy()
    noise_idx = rng.choice(args.n, size=int(0.22 * args.n), replace=False)
    pred_ce[noise_idx] = rng.integers(0, 4, size=len(noise_idx))

    pred_focal = y_true.copy()
    noise_idx = rng.choice(args.n, size=int(0.18 * args.n), replace=False)
    pred_focal[noise_idx] = rng.integers(0, 4, size=len(noise_idx))

    pred_catl = y_true.copy()
    noise_idx = rng.choice(args.n, size=int(0.13 * args.n), replace=False)
    pred_catl[noise_idx] = rng.integers(0, 4, size=len(noise_idx))

    report = compare_models(
        y_true=y_true,
        predictions={
            "cross_entropy_baseline": pred_ce,
            "focal_baseline": pred_focal,
            "catl_proposed": pred_catl,
        },
    )
    out = write_ablation_report(Path(args.output), report)
    print(f"Ablation report written to {out}")


if __name__ == "__main__":
    main()
