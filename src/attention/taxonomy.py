from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class AttentionLevel(str, Enum):
    ENGAGED = "engaged"
    NEUTRAL = "neutral"
    DISTRACTED = "distracted"
    ATTENTION_LOSS = "attention_loss"


@dataclass
class AttentionTaxonomy:
    label_to_level: Dict[str, AttentionLevel]
    level_to_index: Dict[AttentionLevel, int]

    def resolve(self, raw_label: str | None) -> AttentionLevel:
        if not raw_label:
            return AttentionLevel.NEUTRAL
        key = raw_label.strip().lower()
        return self.label_to_level.get(key, AttentionLevel.NEUTRAL)


def default_taxonomy() -> AttentionTaxonomy:
    mapping = {
        # engagement cues
        "curiosity": AttentionLevel.ENGAGED,
        "surprise": AttentionLevel.ENGAGED,
        "concentrated": AttentionLevel.ENGAGED,
        "writing": AttentionLevel.ENGAGED,
        "reading": AttentionLevel.ENGAGED,
        "taking notes": AttentionLevel.ENGAGED,
        "listening": AttentionLevel.ENGAGED,
        # neutral cues
        "neutral": AttentionLevel.NEUTRAL,
        "calm": AttentionLevel.NEUTRAL,
        # disengagement cues
        "boredom": AttentionLevel.DISTRACTED,
        "bored": AttentionLevel.DISTRACTED,
        "sad": AttentionLevel.DISTRACTED,
        "angry": AttentionLevel.DISTRACTED,
        "looking away": AttentionLevel.DISTRACTED,
        "phone": AttentionLevel.DISTRACTED,
        "using phone": AttentionLevel.DISTRACTED,
        # hard attention loss cues
        "sleeping": AttentionLevel.ATTENTION_LOSS,
        "yawning": AttentionLevel.ATTENTION_LOSS,
        "head_down": AttentionLevel.ATTENTION_LOSS,
        "not_present": AttentionLevel.ATTENTION_LOSS,
    }
    level_to_index = {
        AttentionLevel.ENGAGED: 0,
        AttentionLevel.NEUTRAL: 1,
        AttentionLevel.DISTRACTED: 2,
        AttentionLevel.ATTENTION_LOSS: 3,
    }
    return AttentionTaxonomy(label_to_level=mapping, level_to_index=level_to_index)
