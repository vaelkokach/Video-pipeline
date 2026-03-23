from __future__ import annotations

import argparse
from pathlib import Path

from src.data.adapters import DAiSEEAdapter, filter_existing_clips, split_samples
from src.train.runner import TrainConfig, train_attention_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the attention model with CATL.")
    parser.add_argument("--metadata-csv", type=str, required=True, help="CSV with DAiSEE-like schema")
    parser.add_argument("--dataset-root", type=str, default="", help="Root directory for clip paths")
    parser.add_argument("--output-dir", type=str, default="outputs/train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disable-catl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = DAiSEEAdapter(args.metadata_csv)
    samples = adapter.to_attention_samples()
    if args.dataset_root:
        samples = filter_existing_clips(samples, root_dir=args.dataset_root)

    train_samples, val_samples, _ = split_samples(samples)
    cfg = TrainConfig(
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        device=args.device,
        use_catl=not args.disable_catl,
    )
    report = train_attention_model(train_samples=train_samples, val_samples=val_samples, cfg=cfg)
    print("Training complete.")
    print(f"Best val loss ({report['loss_name']}): {report['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
