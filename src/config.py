from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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
    seed: int = 42
    experiment_name: str = "baseline"
    save_run_manifest: bool = True
    anonymized_ids_only: bool = True
    persist_face_crops: bool = False

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


def _apply_dict_overrides(config: PipelineConfig, data: Dict[str, Any]) -> PipelineConfig:
    model_data = data.get("model_paths", {})
    if model_data:
        for key, value in model_data.items():
            if hasattr(config.model_paths, key):
                setattr(config.model_paths, key, value)

    for key, value in data.items():
        if key == "model_paths":
            continue
        if hasattr(config, key):
            if key == "output_dir" and value is not None:
                setattr(config, key, Path(value))
            else:
                setattr(config, key, value)
    return config


def load_config_from_yaml(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML config root must be a mapping/object.")
    return payload


def default_config(config_path: str | None = None) -> PipelineConfig:
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
    config = PipelineConfig(model_paths=model_paths)
    if config_path:
        overrides = load_config_from_yaml(config_path)
        config = _apply_dict_overrides(config, overrides)
    return config