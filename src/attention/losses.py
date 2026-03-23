from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CATLConfig:
    alpha: float = 0.75
    gamma: float = 2.0
    temporal_weight: float = 0.20
    transition_weight: float = 0.15
    transition_margin: float = 0.10
    attention_loss_index: int = 3


class CustomAttentionTransitionLoss(nn.Module):
    """
    CATL: class-balanced focal CE + temporal smoothness + transition margin penalty.

    logits: [B, T, C]
    targets: [B, T] with class indices.
    """

    def __init__(self, config: CATLConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or CATLConfig()

    def _focal_component(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        )
        probs = torch.softmax(logits, dim=-1)
        p_t = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        focal = self.cfg.alpha * (1.0 - p_t).pow(self.cfg.gamma) * ce.reshape_as(targets)
        return focal.mean()

    def _temporal_component(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        if probs.shape[1] < 2:
            return logits.new_tensor(0.0)
        diffs = probs[:, 1:, :] - probs[:, :-1, :]
        return (diffs.pow(2).mean())

    def _transition_component(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)[..., self.cfg.attention_loss_index]
        if targets.shape[1] < 2:
            return logits.new_tensor(0.0)

        prev_t = targets[:, :-1]
        next_t = targets[:, 1:]
        transition_mask = (prev_t != self.cfg.attention_loss_index) & (
            next_t == self.cfg.attention_loss_index
        )
        if not transition_mask.any():
            return logits.new_tensor(0.0)

        prev_probs = probs[:, :-1][transition_mask]
        next_probs = probs[:, 1:][transition_mask]
        margin_violation = F.relu(self.cfg.transition_margin - (next_probs - prev_probs))
        return margin_violation.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self._focal_component(logits, targets)
        temporal_loss = self._temporal_component(logits)
        transition_loss = self._transition_component(logits, targets)
        return (
            focal_loss
            + self.cfg.temporal_weight * temporal_loss
            + self.cfg.transition_weight * transition_loss
        )
