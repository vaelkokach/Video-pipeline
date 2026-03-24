# Attention Analytics

This repository implements a deep learning-based, real-time classroom analytics system for student behavior and attention-loss detection.

Core capabilities:
- student detection and tracking,
- action and emotion inference,
- per-student attention-level estimation and alert events,
- realtime API + dashboard streaming,
- reproducible experiment manifests,
- training/evaluation scaffolding for the thesis contribution.

The primary theoretical contribution is **Custom Attention Transition Loss (CATL)** in `src/attention/losses.py`.

## Quick Start

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

4) Run offline pipeline:

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

5) Optional YAML config:

```powershell
python -m src.main --video path/to/classroom.mp4 --config configs/experiment.example.yaml
```

6) Realtime API + dashboard:

```powershell
python -m src.main_realtime --camera 0 --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Training and Ablation

Train temporal attention model (CATL by default):

```powershell
python scripts/train_attention.py --metadata-csv path/to/daisee_metadata.csv --dataset-root path/to/videos --epochs 10
```

Run ablation harness:

```powershell
python scripts/run_ablation.py --output outputs/eval/ablation_report.json
```

## Thesis Modules

- `src/attention/`:
  - taxonomy, online estimator, event engine, CATL loss, temporal model, calibration.
- `src/data/`:
  - public dataset adapters and label harmonization to thesis taxonomy.
- `src/train/`:
  - training runner and sequence dataset scaffolding.
- `src/eval/`:
  - classification, temporal, calibration metrics and significance tests.
- `src/api/` + `dashboard/`:
  - realtime websocket streaming and instructor-facing dashboard.

## Reproducibility

- deterministic seed control in `src/repro.py`,
- run manifest written to `outputs/run_manifest.json`,
- config override through YAML (`configs/experiment.example.yaml`).

## Privacy and Ethics

See:
- `docs/privacy_ethics.md`
- `docs/theoretical_contribution.md`

## Dependencies

Base dependencies are in `requirements.txt`:
- **Person detection**: YOLOv8 (ultralytics) by default; config `det_config="yolo"`.
- **Action recognition**: MMAction2 with TSM-MobileNetV2 (lightweight, ~2.7M params); run `scripts/download_models.py` first.
- **Emotion recognition**: EmotiEffLib; models auto-download on first use.
- **Realtime service**: FastAPI + WebSocket + dashboard frontend.

## Notes on models

- **Person detection**: YOLOv8n (lightweight).
- **Action recognition**: TSM-MobileNetV2 (~2.7M params, ~3.3G FLOPs). For higher accuracy, switch `config.model_paths` to TSN-R50 in `src/config.py`.
- **Emotion recognition**: EmotiEffLib (default: enet_b0, 8 classes). Use `emotion_model="light"` for mbf_va_mtl (112px, faster).

If your models require different inputs or return different output shapes,
adjust the adapters in `src/`.

## Per-student tracking

Tracking uses a lightweight IOU-based tracker in `src/tracking.py`. For more
robust tracking (occlusions, fast motion), upgrade to DeepSORT/ByteTrack.
