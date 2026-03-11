from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelPaths:
    # MMDetection: config path, or model name (e.g. rtmdet_tiny_8xb32-300e_coco) for auto-download.
    det_config: str
    # MMACTION2 config and checkpoint for action recognition.
    action_config: str
    action_checkpoint: str
    # EmotiEffLib model name (e.g. default, EmotiEffNet-B2).
    emotion_model: str
    # Optional checkpoint override for MMDet; None = use model zoo.
    det_checkpoint: Optional[str] = None


@dataclass
class PipelineConfig:
    model_paths: ModelPaths
    device: str = "cpu"  # Use "cuda:0" if GPU available

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
    # Use "yolo" for YOLOv8 (recommended, no mmcv), or MMDet model name.
    model_paths = ModelPaths(
        det_config="yolo",  # YOLOv8n, auto-downloads; no mmcv compiled ops needed
        det_checkpoint=None,
        action_config="models/mmaction/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py",
        action_checkpoint="models/mmaction/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth",
        emotion_model="default",  # EmotiEffLib default model
    )
    return PipelineConfig(model_paths=model_paths)
