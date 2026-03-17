# Classroom Video Analytics Pipeline

This project analyzes recorded or live classroom video for:
- student count (before analysis starts)
- action recognition via MMAction2
- emotion recognition via EmotiEffLib
- per-student tracking and a final report with summary statistics

The pipeline is intentionally modular: you configure models and weights in
`src/config.py`, and the adapters in `src/` wrap MMAction2, EmotiEffLib, and
your chosen person detector.

## Quick start

1) Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

2) Download MMAction2 models (for action recognition):

```powershell
python scripts/download_models.py
```

3) Fix MMAction2 pip package (DRN module missing):

```powershell
python scripts/fix_mmaction2_drn.py
```

4) Run the pipeline:

```powershell
python -m src.main --video path/to/classroom.mp4 --output-dir outputs
```

To save an annotated video with bounding boxes and per-student labels (student #, action, emotion):

```powershell
python -m src.main --video path/to/classroom.mp4 --output-dir outputs --output-video outputs/annotated.mp4
```

For live feed:

```powershell
python -m src.main --camera 0 --output-dir outputs
```

To limit live feed duration (seconds):

```powershell
python -m src.main --camera 0 --output-dir outputs --max-seconds 30
```

## Dependencies

Base dependencies are in `requirements.txt`:
- **Person detection**: YOLOv8 (ultralytics) by default; config `det_config="yolo"`.
- **Action recognition**: MMAction2 with TSM-MobileNetV2 (lightweight, ~2.7M params); run `scripts/download_models.py` first.
- **Emotion recognition**: EmotiEffLib; models auto-download on first use.

## Notes on models

- **Person detection**: YOLOv8n (lightweight).
- **Action recognition**: TSM-MobileNetV2 (~2.7M params, ~3.3G FLOPs). For higher accuracy, switch `config.model_paths` to TSN-R50 in `src/config.py`.
- **Emotion recognition**: EmotiEffLib (default: enet_b0, 8 classes). Use `emotion_model="light"` for mbf_va_mtl (112px, faster).

If your models require different inputs or return different output shapes,
adjust the adapters in `src/`.

## Per-student tracking

Tracking uses a lightweight IOU-based tracker in `src/tracking.py`. For more
robust tracking (occlusions, fast motion), upgrade to DeepSORT/ByteTrack.
