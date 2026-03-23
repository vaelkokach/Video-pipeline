from __future__ import annotations

import argparse
from pathlib import Path

from .config import default_config
from .pipeline import run_pipeline
from .repro import collect_runtime_manifest, set_global_seed, write_run_manifest
from .video_io import VideoSource


def _config_to_dict(config) -> dict:
    return {
        "device": config.device,
        "seed": config.seed,
        "experiment_name": config.experiment_name,
        "sample_fps": config.sample_fps,
        "anonymized_ids_only": config.anonymized_ids_only,
        "persist_face_crops": config.persist_face_crops,
        "clip_len": config.clip_len,
        "clip_stride": config.clip_stride,
        "crop_size": config.crop_size,
        "resize_width": config.resize_width,
        "resize_height": config.resize_height,
        "count_sample_seconds": config.count_sample_seconds,
        "count_threshold": config.count_threshold,
        "track_iou_threshold": config.track_iou_threshold,
        "track_max_age": config.track_max_age,
        "min_emotion_score": config.min_emotion_score,
        "min_action_score": config.min_action_score,
        "output_dir": str(config.output_dir),
        "model_paths": {
            "det_config": config.model_paths.det_config,
            "det_checkpoint": config.model_paths.det_checkpoint,
            "action_config": config.model_paths.action_config,
            "action_checkpoint": config.model_paths.action_checkpoint,
            "emotion_model": config.model_paths.emotion_model,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classroom video analytics pipeline")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--camera", type=int, help="Camera index (e.g., 0)")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--max-seconds", type=int, help="Stop after N seconds (live feed)")
    parser.add_argument(
        "--output-video",
        type=str,
        help="Save annotated video with boxes and labels (e.g. outputs/annotated.mp4)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional YAML config file. CLI flags still override output/video options.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video and args.camera is None:
        raise SystemExit("Provide --video or --camera.")

    config = default_config(config_path=args.config)
    config.output_dir = Path(args.output_dir)
    output_video = Path(args.output_video) if args.output_video else None
    set_global_seed(config.seed)
    if config.save_run_manifest:
        manifest = collect_runtime_manifest(
            {
                "experiment_name": config.experiment_name,
                "seed": config.seed,
                "config": _config_to_dict(config),
            }
        )
        write_run_manifest(config.output_dir, manifest)

    source = VideoSource(path=args.video, camera_index=args.camera)
    report_path = run_pipeline(
        config, source, max_seconds=args.max_seconds, output_video_path=output_video
    )
    print(f"Report written to {report_path}")
    if output_video:
        print(f"Annotated video written to {output_video}")


if __name__ == "__main__":
    main()
