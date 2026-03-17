from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from .config import PipelineConfig, default_config
from .counting import StudentCounter
from .reporting import ReportStats, write_report
from .tracking import SimpleTracker
from .video_io import VideoSource, iter_sampled_frames, open_capture


def _load_action_recognizer(config: PipelineConfig) -> Optional["ActionRecognizer"]:
    try:
        from .inference_action import ActionRecognizer
        return ActionRecognizer(
            config=str(config.model_paths.action_config),
            checkpoint=str(config.model_paths.action_checkpoint),
            device=config.device,
        )
    except Exception as e:
        print(f"Warning: Action recognition disabled: {e}")
        return None


def _load_emotion_recognizer(config: PipelineConfig) -> Optional["EmotionRecognizer"]:
    try:
        from .inference_emotion import EmotionRecognizer
        return EmotionRecognizer(
            model_name=config.model_paths.emotion_model,
            device=config.device,
        )
    except Exception as e:
        print(f"Warning: Emotion recognition disabled: {e}")
        return None


def _count_students(
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


def _top_prediction(predictions, min_score: float = 0.0) -> Tuple[Optional[str], float]:
    for pred in predictions:
        if pred.score >= min_score:
            return pred.label, pred.score
    return None, 0.0


def _crop_and_resize(frame: np.ndarray, bbox: List[float], size: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (size, size))


def _maybe_resize_frame(frame: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    if width is None or height is None:
        return frame
    return cv2.resize(frame, (width, height))


def _draw_annotations(
    frame: np.ndarray,
    assignments: Dict[int, List[float]],
    current_labels: Dict[int, Dict[str, Optional[str]]],
) -> np.ndarray:
    """Draw bounding boxes and labels (student#, action, emotion) on frame."""
    out = frame.copy()
    for track_id, bbox in assignments.items():
        x1, y1, x2, y2 = [int(v) for v in bbox]
        labels = current_labels.get(track_id, {})
        action = labels.get("action") or "-"
        emotion = labels.get("emotion") or "-"
        text = f"#{track_id} | A:{action} E:{emotion}"
        # Box color (BGR)
        color = (0, 200, 100)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # Label background (ensure it stays in frame)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly1 = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, ly1), (min(x1 + tw + 4, out.shape[1]), y1), color, -1)
        cv2.putText(
            out, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
    return out


def run_pipeline(
    config: PipelineConfig,
    source: VideoSource,
    max_seconds: Optional[int],
    output_video_path: Optional[Path] = None,
) -> Path:
    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source.")

    counter = StudentCounter(
        det_config=str(config.model_paths.det_config),
        det_checkpoint=config.model_paths.det_checkpoint,
        device=config.device,
    )
    action_recognizer = _load_action_recognizer(config)
    emotion_recognizer = _load_emotion_recognizer(config)

    stats = ReportStats()
    stats.sample_fps = config.sample_fps

    # Count students using a short initial window.
    stats.student_count = _count_students(
        cap,
        counter=counter,
        sample_fps=config.sample_fps,
        count_sample_seconds=config.count_sample_seconds,
        score_thr=config.count_threshold,
    )

    # Reopen video file for main loop (seek fails on some codecs; fresh capture ensures full read).
    # Skip for camera - we continue from current position.
    if source.path:
        cap.release()
        cap = open_capture(source)
        if not cap.isOpened():
            raise RuntimeError("Failed to reopen video source for analysis.")

    tracker = SimpleTracker(
        iou_threshold=config.track_iou_threshold,
        max_age=config.track_max_age,
    )
    per_student_clips: Dict[int, List[np.ndarray]] = {}
    current_labels: Dict[int, Dict[str, Optional[str]]] = {}  # track_id -> {action, emotion}
    start_time = time.time()

    video_writer: Optional[cv2.VideoWriter] = None
    if output_video_path:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    for frame_index, frame in iter_sampled_frames(cap, sample_fps=config.sample_fps):
        stats.total_frames += 1
        frame = _maybe_resize_frame(frame, config.resize_width, config.resize_height)

        detections = counter.detect_people(frame, config.count_threshold)
        assignments, expired = tracker.update(detections, frame_index)
        for track_id in expired:
            per_student_clips.pop(track_id, None)
            current_labels.pop(track_id, None)

        for track_id, bbox in assignments.items():
            current_labels.setdefault(track_id, {})
            crop = _crop_and_resize(frame, bbox, size=config.crop_size)
            stats.mark_student_seen(track_id)
            stats.add_student_bbox(track_id, frame_index, bbox)

            # Emotion recognition per student.
            if emotion_recognizer is not None:
                emotion_preds = emotion_recognizer.predict_frame(crop)
                emotion_label, emotion_score = _top_prediction(
                    emotion_preds, min_score=config.min_emotion_score
                )
                if emotion_label:
                    stats.add_emotions([emotion_label])
                    stats.add_student_emotion(track_id, emotion_label)
                    stats.add_student_emotion_score(track_id, emotion_label, emotion_score)
                    current_labels[track_id]["emotion"] = emotion_label

            # Action recognition per student on a rolling clip.
            if action_recognizer is not None:
                clip = per_student_clips.setdefault(track_id, [])
                clip.append(crop)
                if len(clip) >= config.clip_len:
                    clip_array = np.stack(clip[: config.clip_len], axis=0)
                    action_preds = action_recognizer.predict_clip(clip_array)
                    action_label, action_score = _top_prediction(
                        action_preds, min_score=config.min_action_score
                    )
                    if action_label:
                        stats.add_actions([action_label])
                        stats.add_student_action(track_id, action_label)
                        stats.add_student_action_score(track_id, action_label, action_score)
                        current_labels[track_id]["action"] = action_label
                    per_student_clips[track_id] = clip[config.clip_stride :]

        # Write annotated frame to output video
        if output_video_path:
            if video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(output_video_path), fourcc, config.sample_fps, (w, h)
                )
            ann_frame = _draw_annotations(frame, assignments, current_labels)
            video_writer.write(ann_frame)

        if max_seconds is not None and (time.time() - start_time) >= max_seconds:
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    return write_report(stats, config.output_dir)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video and args.camera is None:
        raise SystemExit("Provide --video or --camera.")

    config = default_config()
    config.output_dir = Path(args.output_dir)
    output_video = Path(args.output_video) if args.output_video else None

    source = VideoSource(path=args.video, camera_index=args.camera)
    report_path = run_pipeline(
        config, source, max_seconds=args.max_seconds, output_video_path=output_video
    )
    print(f"Report written to {report_path}")
    if output_video:
        print(f"Annotated video written to {output_video}")


if __name__ == "__main__":
    main()
