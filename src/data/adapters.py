from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from ..attention.taxonomy import AttentionLevel
from .taxonomy_mapping import map_public_label_to_attention


@dataclass
class EngagementRecord:
    clip_path: str
    raw_label: str
    student_id: str
    split: str


@dataclass
class AttentionSample:
    clip_path: str
    level: AttentionLevel
    level_index: int
    student_id: str
    split: str
    source_dataset: str


LEVEL_TO_INDEX = {
    AttentionLevel.ENGAGED: 0,
    AttentionLevel.NEUTRAL: 1,
    AttentionLevel.DISTRACTED: 2,
    AttentionLevel.ATTENTION_LOSS: 3,
}


class GenericCSVAdapter:
    """Reads public dataset metadata and maps labels into thesis taxonomy."""

    def __init__(
        self,
        csv_path: str | Path,
        dataset_name: str,
        clip_col: str = "clip_path",
        label_col: str = "label",
        split_col: str = "split",
        student_col: str = "student_id",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.dataset_name = dataset_name
        self.clip_col = clip_col
        self.label_col = label_col
        self.split_col = split_col
        self.student_col = student_col

    def records(self) -> Iterable[EngagementRecord]:
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield EngagementRecord(
                    clip_path=str(row[self.clip_col]),
                    raw_label=str(row[self.label_col]),
                    student_id=str(row.get(self.student_col, "unknown")),
                    split=str(row.get(self.split_col, "train")),
                )

    def to_attention_samples(self) -> List[AttentionSample]:
        out: List[AttentionSample] = []
        for item in self.records():
            level = map_public_label_to_attention(item.raw_label)
            out.append(
                AttentionSample(
                    clip_path=item.clip_path,
                    level=level,
                    level_index=LEVEL_TO_INDEX[level],
                    student_id=item.student_id,
                    split=item.split,
                    source_dataset=self.dataset_name,
                )
            )
        return out


class DAiSEEAdapter(GenericCSVAdapter):
    """
    DAiSEE adapter expects a CSV prepared with columns:
    - clip_path
    - engagement_label (very_low/low/medium/high)
    - split
    - student_id
    """

    def __init__(self, csv_path: str | Path) -> None:
        super().__init__(
            csv_path=csv_path,
            dataset_name="DAiSEE",
            clip_col="clip_path",
            label_col="engagement_label",
            split_col="split",
            student_col="student_id",
        )


def split_samples(
    samples: List[AttentionSample],
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
) -> tuple[List[AttentionSample], List[AttentionSample], List[AttentionSample]]:
    train: List[AttentionSample] = []
    val: List[AttentionSample] = []
    test: List[AttentionSample] = []

    for sample in samples:
        split_key = sample.split.lower()
        if split_key == train_split:
            train.append(sample)
        elif split_key == val_split:
            val.append(sample)
        elif split_key == test_split:
            test.append(sample)
    return train, val, test


def filter_existing_clips(samples: List[AttentionSample], root_dir: str | Path | None = None) -> List[AttentionSample]:
    root = Path(root_dir) if root_dir is not None else None
    filtered: List[AttentionSample] = []
    for sample in samples:
        clip_path = Path(sample.clip_path)
        if not clip_path.is_absolute() and root is not None:
            clip_path = root / clip_path
        if clip_path.exists():
            filtered.append(sample)
    return filtered
