from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import json
import math


@dataclass
class ReportStats:
    student_count: int = 0
    total_frames: int = 0
    sample_fps: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)
    emotion_counts: Dict[str, int] = field(default_factory=dict)
    per_student: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    per_student_meta: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def add_actions(self, labels: List[str]) -> None:
        for label in labels:
            self.action_counts[label] = self.action_counts.get(label, 0) + 1

    def add_emotions(self, labels: List[str]) -> None:
        for label in labels:
            self.emotion_counts[label] = self.emotion_counts.get(label, 0) + 1

    def add_student_action(self, student_id: int, label: str) -> None:
        student = self.per_student.setdefault(str(student_id), {"actions": {}, "emotions": {}})
        actions = student["actions"]
        actions[label] = actions.get(label, 0) + 1
        meta = self.per_student_meta.setdefault(
            str(student_id),
            {
                "frames_seen": 0,
                "action_scores": {},
                "emotion_scores": {},
                "action_total": 0,
                "emotion_total": 0,
                "action_change_count": 0,
                "emotion_change_count": 0,
                "last_action": None,
                "last_emotion": None,
            },
        )
        meta["action_total"] = int(meta["action_total"]) + 1
        last_action = meta["last_action"]
        if last_action is not None and last_action != label:
            meta["action_change_count"] = int(meta["action_change_count"]) + 1
        meta["last_action"] = label

    def add_student_action_score(self, student_id: int, label: str, score: float) -> None:
        meta = self.per_student_meta.setdefault(
            str(student_id),
            {
                "frames_seen": 0,
                "action_scores": {},
                "emotion_scores": {},
                "action_total": 0,
                "emotion_total": 0,
                "action_change_count": 0,
                "emotion_change_count": 0,
                "last_action": None,
                "last_emotion": None,
            },
        )
        action_scores = meta["action_scores"]
        action_scores[label] = float(action_scores.get(label, 0.0)) + float(score)

    def add_student_emotion(self, student_id: int, label: str) -> None:
        student = self.per_student.setdefault(str(student_id), {"actions": {}, "emotions": {}})
        emotions = student["emotions"]
        emotions[label] = emotions.get(label, 0) + 1
        meta = self.per_student_meta.setdefault(
            str(student_id),
            {
                "frames_seen": 0,
                "action_scores": {},
                "emotion_scores": {},
                "action_total": 0,
                "emotion_total": 0,
                "action_change_count": 0,
                "emotion_change_count": 0,
                "last_action": None,
                "last_emotion": None,
            },
        )
        meta["emotion_total"] = int(meta["emotion_total"]) + 1
        last_emotion = meta["last_emotion"]
        if last_emotion is not None and last_emotion != label:
            meta["emotion_change_count"] = int(meta["emotion_change_count"]) + 1
        meta["last_emotion"] = label

    def add_student_emotion_score(self, student_id: int, label: str, score: float) -> None:
        meta = self.per_student_meta.setdefault(
            str(student_id),
            {
                "frames_seen": 0,
                "action_scores": {},
                "emotion_scores": {},
                "action_total": 0,
                "emotion_total": 0,
                "action_change_count": 0,
                "emotion_change_count": 0,
                "last_action": None,
                "last_emotion": None,
            },
        )
        emotion_scores = meta["emotion_scores"]
        emotion_scores[label] = float(emotion_scores.get(label, 0.0)) + float(score)

    def mark_student_seen(self, student_id: int) -> None:
        meta = self.per_student_meta.setdefault(
            str(student_id),
            {
                "frames_seen": 0,
                "action_scores": {},
                "emotion_scores": {},
                "action_total": 0,
                "emotion_total": 0,
                "action_change_count": 0,
                "emotion_change_count": 0,
                "last_action": None,
                "last_emotion": None,
            },
        )
        meta["frames_seen"] = int(meta["frames_seen"]) + 1


def _sorted_counts(counts: Dict[str, int]) -> Dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _dominant_label(counts: Dict[str, int]) -> str | None:
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p + 1e-9)
    return float(ent)


def _rate_per_minute(counts: Dict[str, int], total_minutes: float) -> Dict[str, float]:
    if total_minutes <= 0:
        return {label: 0.0 for label in counts}
    return {label: count / total_minutes for label, count in counts.items()}


def _top_ratio(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    top = max(counts.values())
    return float(top / total)


def _avg_scores(score_sums: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    averages: Dict[str, float] = {}
    for label, total_score in score_sums.items():
        count = counts.get(label, 0)
        if count > 0:
            averages[label] = total_score / count
    return averages


def write_report(stats: ReportStats, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"

    total_seconds = 0.0
    if stats.sample_fps > 0:
        total_seconds = stats.total_frames / stats.sample_fps
    total_minutes = total_seconds / 60.0 if total_seconds > 0 else 0.0

    per_student_metrics = {}
    for student_id, payload in stats.per_student.items():
        actions = payload.get("actions", {})
        emotions = payload.get("emotions", {})
        meta = stats.per_student_meta.get(student_id, {})

        frames_seen = int(meta.get("frames_seen", 0))
        action_total = int(meta.get("action_total", 0))
        emotion_total = int(meta.get("emotion_total", 0))
        action_change_count = int(meta.get("action_change_count", 0))
        emotion_change_count = int(meta.get("emotion_change_count", 0))
        action_score_sums = meta.get("action_scores", {})
        emotion_score_sums = meta.get("emotion_scores", {})

        coverage_seconds = frames_seen / stats.sample_fps if stats.sample_fps > 0 else 0.0
        detection_rate = frames_seen / stats.total_frames if stats.total_frames > 0 else 0.0

        per_student_metrics[student_id] = {
            "dominant_action": _dominant_label(actions),
            "dominant_emotion": _dominant_label(emotions),
            "action_diversity": len(actions),
            "emotion_diversity": len(emotions),
            "action_entropy": _entropy(actions),
            "emotion_entropy": _entropy(emotions),
            "action_rate_per_minute": _rate_per_minute(actions, total_minutes),
            "emotion_rate_per_minute": _rate_per_minute(emotions, total_minutes),
            "action_top_ratio": _top_ratio(actions),
            "emotion_top_ratio": _top_ratio(emotions),
            "action_change_rate_per_min": action_change_count / total_minutes if total_minutes > 0 else 0.0,
            "emotion_change_rate_per_min": (
                emotion_change_count / total_minutes if total_minutes > 0 else 0.0
            ),
            "coverage_seconds": coverage_seconds,
            "detection_rate": detection_rate,
            "avg_action_confidence": (
                sum(action_score_sums.values()) / action_total if action_total > 0 else 0.0
            ),
            "avg_emotion_confidence": (
                sum(emotion_score_sums.values()) / emotion_total if emotion_total > 0 else 0.0
            ),
            "avg_action_confidence_per_label": _avg_scores(action_score_sums, actions),
            "avg_emotion_confidence_per_label": _avg_scores(emotion_score_sums, emotions),
        }

    payload = {
        "student_count": stats.student_count,
        "total_frames": stats.total_frames,
        "sample_fps": stats.sample_fps,
        "total_seconds": total_seconds,
        "action_counts": _sorted_counts(stats.action_counts),
        "emotion_counts": _sorted_counts(stats.emotion_counts),
        "per_student": stats.per_student,
        "per_student_metrics": per_student_metrics,
    }

    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path
