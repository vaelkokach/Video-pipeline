from __future__ import annotations

import statistics
import time
from typing import List, Optional, Tuple

from ..config import PipelineConfig
from ..counting import StudentCounter
from ..video_io import iter_sampled_frames


def load_action_recognizer(config: PipelineConfig):
    try:
        from ..inference_action import ActionRecognizer

        return ActionRecognizer(
            config=str(config.model_paths.action_config),
            checkpoint=str(config.model_paths.action_checkpoint),
            device=config.device,
        )
    except Exception as exc:
        print(f"Warning: Action recognition disabled: {exc}")
        return None


def load_emotion_recognizer(config: PipelineConfig):
    try:
        from ..inference_emotion import EmotionRecognizer

        return EmotionRecognizer(
            model_name=config.model_paths.emotion_model,
            device=config.device,
        )
    except Exception as exc:
        print(f"Warning: Emotion recognition disabled: {exc}")
        return None


def count_students(
    cap,
    counter: StudentCounter,
    sample_fps: int,
    count_sample_seconds: int,
    score_thr: float,
) -> int:
    counts: List[int] = []
    start_time = time.time()

    for _, frame in iter_sampled_frames(cap, sample_fps=sample_fps):
        counts.append(counter.count_people(frame, score_thr))
        if time.time() - start_time >= count_sample_seconds:
            break

    if not counts:
        return 0
    return int(statistics.median(counts))


def top_prediction(predictions, min_score: float = 0.0) -> Tuple[Optional[str], float]:
    for pred in predictions:
        if pred.score >= min_score:
            return pred.label, pred.score
    return None, 0.0
