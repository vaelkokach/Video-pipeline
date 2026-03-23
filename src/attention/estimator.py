from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .taxonomy import AttentionLevel, AttentionTaxonomy, default_taxonomy


@dataclass
class AttentionObservation:
    student_id: int
    bbox: List[float]
    frame_width: int
    frame_height: int
    action_label: Optional[str]
    emotion_label: Optional[str]
    action_score: float = 0.0
    emotion_score: float = 0.0


@dataclass
class AttentionOutput:
    student_id: int
    attention_score: float
    loss_probability: float
    level: AttentionLevel
    cue_breakdown: dict


class AttentionEstimator:
    """
    Rule-informed estimator used online for real-time operation.
    A trainable deep model can replace this estimator for offline training and deployment.
    """

    def __init__(
        self,
        taxonomy: AttentionTaxonomy | None = None,
        gaze_weight: float = 0.35,
        posture_weight: float = 0.25,
        emotion_weight: float = 0.20,
        action_weight: float = 0.20,
    ) -> None:
        self.taxonomy = taxonomy or default_taxonomy()
        self.gaze_weight = gaze_weight
        self.posture_weight = posture_weight
        self.emotion_weight = emotion_weight
        self.action_weight = action_weight

    def _posture_signal(self, bbox: List[float], frame_size: Tuple[int, int]) -> float:
        x1, y1, x2, y2 = bbox
        frame_w, frame_h = frame_size
        center_x = ((x1 + x2) / 2.0) / max(frame_w, 1)
        center_y = ((y1 + y2) / 2.0) / max(frame_h, 1)
        height = max(1.0, y2 - y1)
        width = max(1.0, x2 - x1)
        aspect = height / width

        center_bonus = 1.0 - min(abs(center_x - 0.5) * 1.5, 1.0)
        upright_bonus = min(max((aspect - 1.2) / 1.8, 0.0), 1.0)
        lean_penalty = 1.0 - min(abs(center_y - 0.5) * 1.8, 1.0)
        return max(0.0, min(1.0, 0.5 * center_bonus + 0.35 * upright_bonus + 0.15 * lean_penalty))

    def _gaze_proxy_signal(self, action_label: str | None) -> float:
        if not action_label:
            return 0.5
        label = action_label.lower()
        positives = ("reading", "writing", "listening", "taking notes", "studying")
        negatives = ("phone", "texting", "looking away", "sleeping", "gaming")
        if any(token in label for token in positives):
            return 0.85
        if any(token in label for token in negatives):
            return 0.15
        return 0.5

    def _attention_from_level(self, level: AttentionLevel) -> float:
        if level == AttentionLevel.ENGAGED:
            return 0.9
        if level == AttentionLevel.NEUTRAL:
            return 0.6
        if level == AttentionLevel.DISTRACTED:
            return 0.35
        return 0.1

    def predict(self, observation: AttentionObservation) -> AttentionOutput:
        emotion_level = self.taxonomy.resolve(observation.emotion_label)
        action_level = self.taxonomy.resolve(observation.action_label)

        posture = self._posture_signal(
            observation.bbox, (observation.frame_width, observation.frame_height)
        )
        gaze = self._gaze_proxy_signal(observation.action_label)
        emotion_signal = self._attention_from_level(emotion_level)
        action_signal = self._attention_from_level(action_level)

        weighted_score = (
            self.gaze_weight * gaze
            + self.posture_weight * posture
            + self.emotion_weight * emotion_signal
            + self.action_weight * action_signal
        )
        attention_score = max(0.0, min(1.0, weighted_score))
        loss_probability = 1.0 - attention_score

        if attention_score >= 0.75:
            level = AttentionLevel.ENGAGED
        elif attention_score >= 0.5:
            level = AttentionLevel.NEUTRAL
        elif attention_score >= 0.3:
            level = AttentionLevel.DISTRACTED
        else:
            level = AttentionLevel.ATTENTION_LOSS

        return AttentionOutput(
            student_id=observation.student_id,
            attention_score=attention_score,
            loss_probability=loss_probability,
            level=level,
            cue_breakdown={
                "gaze_proxy": gaze,
                "posture_proxy": posture,
                "emotion_signal": emotion_signal,
                "action_signal": action_signal,
            },
        )
