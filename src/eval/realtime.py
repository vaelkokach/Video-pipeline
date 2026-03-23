from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class RealtimeSample:
    frame_latency_ms: float
    fps: float
    cpu_percent: float
    gpu_percent: float


def summarize_realtime(samples: List[RealtimeSample]) -> Dict[str, float]:
    if not samples:
        return {
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_fps": 0.0,
            "avg_cpu_percent": 0.0,
            "avg_gpu_percent": 0.0,
        }
    lat = np.array([s.frame_latency_ms for s in samples], dtype=np.float32)
    fps = np.array([s.fps for s in samples], dtype=np.float32)
    cpu = np.array([s.cpu_percent for s in samples], dtype=np.float32)
    gpu = np.array([s.gpu_percent for s in samples], dtype=np.float32)
    return {
        "avg_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "avg_fps": float(fps.mean()),
        "avg_cpu_percent": float(cpu.mean()),
        "avg_gpu_percent": float(gpu.mean()),
    }
