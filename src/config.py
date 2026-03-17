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
    # EmotiEffLib model (default=enet_b0, light=mbf_va_mtl 112px, better=enet_b2_8).
    emotion_model: str
    # Optional checkpoint override for MMDet; None = use model zoo.
    det_checkpoint: Optional[str] = None


@dataclass
class PipelineConfig:
    model_paths: ModelPaths
    device: str = "cpu"  # Use "cuda:0" if GPU available

    # Video sampling
    sample_fps: int = 2
    clip_len: int = 8  # TSN needs 8 segments; 8 frames at 2fps = 4 sec to first action
    clip_stride: int = 4
    crop_size: int = 224
    resize_width: int | None = None
    resize_height: int | None = None

    # Counting
    count_sample_seconds: int = 10
    count_threshold: float = 0.25  # Lower = more detections (distant/small students); raise if too many false positives

    # Tracking
    track_iou_threshold: float = 0.3
    track_max_age: int = 30

    # Scoring thresholds
    min_emotion_score: float = 0.0
    min_action_score: float = 0.0

    # Reporting
    output_dir: Path = Path("outputs")


def default_config() -> PipelineConfig:
    # Best models in EmotiEffLib and MMAction2 (8-frame clips).
    # Action: TSN-R50 ~73% (best for 8 frames); lightweight: TSM-MobileNetV2 ~69%.
    # Emotion: enet_b2_8 ~63% AffectNet (best); lighter: default=enet_b0, light=mbf_va_mtl.
    model_paths = ModelPaths(
        det_config="yolov8s",  # yolov8s=small (better detection); yolo/yolov8n=nano (faster)
        det_checkpoint=None,
        action_config="models/mmaction/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py",
        action_checkpoint="models/mmaction/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth",
        emotion_model="enet_b2_8",
    )
    return PipelineConfig(model_paths=model_paths)