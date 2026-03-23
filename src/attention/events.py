from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .taxonomy import AttentionLevel


@dataclass
class AttentionEvent:
    student_id: int
    level: AttentionLevel
    attention_score: float
    loss_probability: float
    event_type: str


class AttentionEventEngine:
    def __init__(self, alert_threshold: float = 0.70, cooldown_frames: int = 8) -> None:
        self.alert_threshold = alert_threshold
        self.cooldown_frames = cooldown_frames
        self._last_alert_frame: Dict[int, int] = {}
        self._last_level: Dict[int, AttentionLevel] = {}

    def update(
        self,
        student_id: int,
        frame_index: int,
        level: AttentionLevel,
        attention_score: float,
        loss_probability: float,
    ) -> Optional[AttentionEvent]:
        previous = self._last_level.get(student_id)
        self._last_level[student_id] = level

        if previous is not None and previous != level:
            return AttentionEvent(
                student_id=student_id,
                level=level,
                attention_score=attention_score,
                loss_probability=loss_probability,
                event_type="level_transition",
            )

        last_alert = self._last_alert_frame.get(student_id, -10_000)
        if loss_probability >= self.alert_threshold and (frame_index - last_alert) >= self.cooldown_frames:
            self._last_alert_frame[student_id] = frame_index
            return AttentionEvent(
                student_id=student_id,
                level=level,
                attention_score=attention_score,
                loss_probability=loss_probability,
                event_type="attention_loss_alert",
            )
        return None
