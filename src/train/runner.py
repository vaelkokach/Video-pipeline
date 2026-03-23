from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from ..attention.losses import CATLConfig, CustomAttentionTransitionLoss
from ..attention.model import TemporalAttentionFusionModel
from ..data.adapters import AttentionSample
from ..repro import set_global_seed
from .datasets import AttentionSequenceDataset, SequenceBuildConfig


@dataclass
class TrainConfig:
    output_dir: Path = Path("outputs/train")
    seed: int = 42
    batch_size: int = 8
    epochs: int = 5
    lr: float = 1e-3
    feature_dim: int = 16
    seq_len: int = 16
    num_classes: int = 4
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    device: str = "cpu"
    use_catl: bool = True


def _train_one_epoch(
    model: torch.nn.Module,
    loss_fn,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    losses: List[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loss_fn, loader: DataLoader, device: str) -> float:
    model.eval()
    losses: List[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


def train_attention_model(
    train_samples: List[AttentionSample],
    val_samples: List[AttentionSample],
    cfg: TrainConfig | None = None,
) -> Dict[str, object]:
    cfg = cfg or TrainConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(cfg.seed)

    build_cfg = SequenceBuildConfig(seq_len=cfg.seq_len, feature_dim=cfg.feature_dim)
    train_ds = AttentionSequenceDataset(train_samples, build_cfg)
    val_ds = AttentionSequenceDataset(val_samples, build_cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TemporalAttentionFusionModel(
        input_dim=cfg.feature_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_classes=cfg.num_classes,
    ).to(cfg.device)

    if cfg.use_catl:
        loss_fn = CustomAttentionTransitionLoss(CATLConfig())
        loss_name = "CATL"
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

        def _ce_wrapper(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        loss_fn = _ce_wrapper
        loss_name = "CrossEntropy"

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: List[Dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        train_loss = _train_one_epoch(model, loss_fn, train_loader, optimizer, cfg.device)
        val_loss = _evaluate(model, loss_fn, val_loader, cfg.device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.output_dir / "best_model.pt")

    report = {
        "loss_name": loss_name,
        "best_val_loss": best_val,
        "history": history,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
    }
    (cfg.output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
