# Classroom Video Analytics Pipeline

This project analyzes recorded or live classroom video for:
- student count (before analysis starts)
- action recognition via MMACTION2
- emotion recognition via EmotiEffLib
- per-student tracking and a final report with summary statistics

The pipeline is intentionally modular: you configure models and weights in
`src/config.py`, and the adapters in `src/` wrap MMACTION2, EmotiEffLib, and
your chosen person detector.

## Quick start

1) Create a virtual environment (recommended) and install dependencies.
2) Install MMACTION2 and EmotiEffLib following their official instructions.
3) Download model weights and update paths in `src/config.py`.
4) Run the pipeline:

```bash
python -m src.main --video path/to/classroom.mp4 --output-dir outputs
```

For live feed:

```bash
python -m src.main --camera 0 --output-dir outputs
```

## Dependencies

Base dependencies are in `requirements.txt`. You will still need to install
MMACTION2 and EmotiEffLib from source or pip as instructed by their projects.

## Notes on models

- **Student counting**: uses an MMDetection model (person class).
- **Action recognition**: uses MMACTION2 video-level inference.
- **Emotion recognition**: uses EmotiEffLib on detected faces or full frames.

If your models require different inputs or return different output shapes,
adjust the adapters in `src/`.

## Per-student tracking

Tracking uses a lightweight IOU-based tracker in `src/tracking.py`. For more
robust tracking (occlusions, fast motion), upgrade to DeepSORT/ByteTrack.
