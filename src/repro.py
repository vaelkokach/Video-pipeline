from __future__ import annotations

import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Keep CPU-only fallback compatible when torch is unavailable.
        pass


def collect_runtime_manifest(extra: Dict[str, object] | None = None) -> Dict[str, object]:
    manifest: Dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        manifest["torch_version"] = None
        manifest["cuda_available"] = False

    if extra:
        manifest.update(extra)
    return manifest


def write_run_manifest(output_dir: Path, manifest: Dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
