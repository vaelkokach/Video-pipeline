from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from ..config import PipelineConfig
from ..counting import StudentCounter
from ..reporting import ReportStats, write_report
from ..tracking import SimpleTracker
from ..video_io import VideoSource, iter_sampled_frames, open_capture
from ..attention.estimator import AttentionEstimator, AttentionObservation
from ..attention.events import AttentionEventEngine
from .components import count_students, load_action_recognizer, load_emotion_recognizer, top_prediction
from .frame_ops import crop_and_resize, draw_annotations, maybe_resize_frame


def run_pipeline(
    config: PipelineConfig,
    source: VideoSource,
    max_seconds: Optional[int],
    output_video_path: Optional[Path] = None,
    realtime_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Path:
    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source.")

    counter = StudentCounter(
        det_config=str(config.model_paths.det_config),
        det_checkpoint=config.model_paths.det_checkpoint,
        device=config.device,
    )
    action_recognizer = load_action_recognizer(config)
    emotion_recognizer = load_emotion_recognizer(config)

    stats = ReportStats()
    stats.sample_fps = config.sample_fps

    stats.student_count = count_students(
        cap,
        counter=counter,
        sample_fps=config.sample_fps,
        count_sample_seconds=config.count_sample_seconds,
        score_thr=config.count_threshold,
    )

    if source.path:
        cap.release()
        cap = open_capture(source)
        if not cap.isOpened():
            raise RuntimeError("Failed to reopen video source for analysis.")

    tracker = SimpleTracker(
        iou_threshold=config.track_iou_threshold,
        max_age=config.track_max_age,
    )
    attention_estimator = AttentionEstimator()
    event_engine = AttentionEventEngine()
    per_student_clips: Dict[int, List[np.ndarray]] = {}
    current_labels: Dict[int, Dict[str, Optional[str]]] = {}
    start_time = time.time()

    video_writer: Optional[cv2.VideoWriter] = None
    if output_video_path:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    for frame_index, frame in iter_sampled_frames(cap, sample_fps=config.sample_fps):
        stats.total_frames += 1
        frame = maybe_resize_frame(frame, config.resize_width, config.resize_height)
        frame_h, frame_w = frame.shape[:2]
        frame_payload: Dict[str, object] = {"frame_index": frame_index, "students": [], "events": []}

        detections = counter.detect_people(frame, config.count_threshold)
        assignments, expired = tracker.update(detections, frame_index)
        for track_id in expired:
            per_student_clips.pop(track_id, None)
            current_labels.pop(track_id, None)

        for track_id, bbox in assignments.items():
            current_labels.setdefault(track_id, {})
            crop = crop_and_resize(frame, bbox, size=config.crop_size)
            stats.mark_student_seen(track_id)
            stats.add_student_bbox(track_id, frame_index, bbox)
            action_label: Optional[str] = current_labels[track_id].get("action")
            emotion_label: Optional[str] = current_labels[track_id].get("emotion")
            action_score = 0.0
            emotion_score = 0.0

            if emotion_recognizer is not None:
                emotion_preds = emotion_recognizer.predict_frame(crop)
                emotion_label, emotion_score = top_prediction(
                    emotion_preds, min_score=config.min_emotion_score
                )
                if emotion_label:
                    stats.add_emotions([emotion_label])
                    stats.add_student_emotion(track_id, emotion_label)
                    stats.add_student_emotion_score(track_id, emotion_label, emotion_score)
                    current_labels[track_id]["emotion"] = emotion_label

            if action_recognizer is not None:
                clip = per_student_clips.setdefault(track_id, [])
                clip.append(crop)
                if len(clip) >= config.clip_len:
                    clip_array = np.stack(clip[: config.clip_len], axis=0)
                    action_preds = action_recognizer.predict_clip(clip_array)
                    action_label, action_score = top_prediction(
                        action_preds, min_score=config.min_action_score
                    )
                    if action_label:
                        stats.add_actions([action_label])
                        stats.add_student_action(track_id, action_label)
                        stats.add_student_action_score(track_id, action_label, action_score)
                        current_labels[track_id]["action"] = action_label
                    per_student_clips[track_id] = clip[config.clip_stride :]

            attention_output = attention_estimator.predict(
                AttentionObservation(
                    student_id=track_id,
                    bbox=bbox,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    action_label=current_labels[track_id].get("action"),
                    emotion_label=current_labels[track_id].get("emotion"),
                    action_score=action_score,
                    emotion_score=emotion_score,
                )
            )
            stats.add_attention_levels([attention_output.level.value])
            stats.add_student_attention(
                track_id,
                attention_output.level.value,
                attention_output.attention_score,
                attention_output.loss_probability,
            )
            attention_event = event_engine.update(
                student_id=track_id,
                frame_index=frame_index,
                level=attention_output.level,
                attention_score=attention_output.attention_score,
                loss_probability=attention_output.loss_probability,
            )
            student_payload = {
                "student_id": track_id,
                "bbox": [float(v) for v in bbox],
                "action": current_labels[track_id].get("action"),
                "emotion": current_labels[track_id].get("emotion"),
                "attention_level": attention_output.level.value,
                "attention_score": float(attention_output.attention_score),
                "loss_probability": float(attention_output.loss_probability),
                "cue_breakdown": attention_output.cue_breakdown,
            }
            frame_payload["students"].append(student_payload)
            if attention_event is not None:
                frame_payload["events"].append(
                    {
                        "student_id": attention_event.student_id,
                        "event_type": attention_event.event_type,
                        "level": attention_event.level.value,
                        "attention_score": attention_event.attention_score,
                        "loss_probability": attention_event.loss_probability,
                    }
                )

        if output_video_path:
            if video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(output_video_path), fourcc, config.sample_fps, (w, h)
                )
            ann_frame = draw_annotations(frame, assignments, current_labels)
            video_writer.write(ann_frame)

        if realtime_callback is not None:
            frame_payload["student_count"] = len(frame_payload["students"])
            realtime_callback(frame_payload)

        if max_seconds is not None and (time.time() - start_time) >= max_seconds:
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    return write_report(stats, config.output_dir)
