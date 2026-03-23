from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.adapters import AttentionSample


@dataclass
class SequenceBuildConfig:
    seq_len: int = 16
    feature_dim: int = 16


def _synthetic_features_for_sample(sample: AttentionSample, seq_len: int, feature_dim: int) -> np.ndarray:
    """
    Placeholder feature extraction for training scaffolding.
    Replace with true multimodal feature extraction in production experiments.
    """
    rng = np.random.default_rng(abs(hash((sample.clip_path, sample.student_id))) % (2**32))
    base = rng.normal(loc=0.0, scale=1.0, size=(seq_len, feature_dim)).astype(np.float32)
    level_bias = float(sample.level_index) / 3.0
    base[:, 0] = np.clip(1.0 - level_bias + 0.1 * base[:, 0], 0.0, 1.0)  # engagement proxy
    base[:, 1] = np.clip(level_bias + 0.1 * base[:, 1], 0.0, 1.0)  # attention-loss proxy
    return base


class AttentionSequenceDataset(Dataset):
    def __init__(self, samples: List[AttentionSample], cfg: SequenceBuildConfig | None = None) -> None:
        self.samples = samples
        self.cfg = cfg or SequenceBuildConfig()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        features = _synthetic_features_for_sample(
            sample=sample, seq_len=self.cfg.seq_len, feature_dim=self.cfg.feature_dim
        )
        targets = np.full((self.cfg.seq_len,), fill_value=sample.level_index, dtype=np.int64)
        return torch.from_numpy(features), torch.from_numpy(targets)
