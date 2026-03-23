from __future__ import annotations

from typing import Dict

from ..attention.taxonomy import AttentionLevel


PUBLIC_LABEL_MAP: Dict[str, AttentionLevel] = {
    # DAiSEE engagement levels (0-3) mapped via string labels
    "very_low": AttentionLevel.ATTENTION_LOSS,
    "low": AttentionLevel.DISTRACTED,
    "medium": AttentionLevel.NEUTRAL,
    "high": AttentionLevel.ENGAGED,
    # Generic/public emotion synonyms
    "boredom": AttentionLevel.DISTRACTED,
    "confusion": AttentionLevel.NEUTRAL,
    "engagement": AttentionLevel.ENGAGED,
    "frustration": AttentionLevel.DISTRACTED,
}


def map_public_label_to_attention(label: str) -> AttentionLevel:
    key = label.strip().lower()
    return PUBLIC_LABEL_MAP.get(key, AttentionLevel.NEUTRAL)
