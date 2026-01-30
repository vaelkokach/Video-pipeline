from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelPaths:
    # MMDetection config and checkpoint for person detection.
    det_config: Path
    det_checkpoint: Path

    # MMACTION2 config and checkpoint for action recognition.
    action_config: Path
    action_checkpoint: Path

    # EmotiEffLib model name or checkpoint (depends on library usage).
    emotion_model: str


@dataclass
class PipelineConfig:
    model_paths: ModelPaths
    device: str = "cuda:0"

    # Video sampling
    sample_fps: int = 2
    clip_len: int = 32
    clip_stride: int = 16
    crop_size: int = 224
    resize_width: int | None = None
    resize_height: int | None = None

    # Counting
    count_sample_seconds: int = 10
    count_threshold: float = 0.4

    # Tracking
    track_iou_threshold: float = 0.3
    track_max_age: int = 30

    # Scoring thresholds
    min_emotion_score: float = 0.0
    min_action_score: float = 0.0

    # Reporting
    output_dir: Path = Path("outputs")


def default_config() -> PipelineConfig:
    # Update these paths for your environment.
    model_paths = ModelPaths(
        det_config=Path("models/mmdet/person_det.py"),
        det_checkpoint=Path("models/mmdet/person_det.pth"),
        action_config=Path("models/mmaction/action.py"),
        action_checkpoint=Path("models/mmaction/action.pth"),
        emotion_model="emotiefflib_default",
    )
    return PipelineConfig(model_paths=model_paths)
