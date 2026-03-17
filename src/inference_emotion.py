from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List

import numpy as np


def _patch_emotiefflib_download() -> None:
    """Patch EmotiEffLib download to use requests with retries (avoids 429 rate limit)."""
    import urllib.request

    try:
        import requests
    except ImportError:
        return  # Fall back to default behavior

    _orig_urlretrieve = urllib.request.urlretrieve

    def _robust_download(url: str, path: str, max_retries: int = 5) -> None:
        for attempt in range(max_retries):
            try:
                r = requests.get(
                    url,
                    headers={"User-Agent": "EmotiEffLib/1.1.1 (video-pipeline)"},
                    timeout=60,
                    stream=True,
                )
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"Download retry {attempt + 1}/{max_retries} in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise

    def _patched_urlretrieve(url: str, filename: str | None = None, *_args, **_kwargs):
        if filename and "github.com" in url:
            _robust_download(url, filename)
            return filename, None
        return _orig_urlretrieve(url, filename, *_args, **_kwargs)

    urllib.request.urlretrieve = _patched_urlretrieve

    # Patch emotiefflib.utils.download_model to use requests if available
    import emotiefflib.utils as emotieff_utils

    _orig_download = emotieff_utils.download_model

    def _is_valid_model(path: str, model_file: str) -> bool:
        """Check if downloaded file is valid (not HTML error page)."""
        if not os.path.isfile(path) or os.path.getsize(path) < 50_000:
            return False
        with open(path, "rb") as f:
            head = f.read(100)
        # HTML or error page starts with < or contains "html"
        if head.startswith(b"<") or b"html" in head.lower() or b"429" in head:
            return False
        if model_file.endswith(".pt"):
            return head[:2] == b"PK" or head[:4] == b"\x80\x02"  # zip or pickle
        return True  # .onnx, .h5 - basic size check suffices

    def _robust_download_model(model_file: str, path_in_repo: str) -> str:
        cache_dir = os.path.join(os.path.expanduser("~"), ".emotiefflib")
        os.makedirs(cache_dir, exist_ok=True)
        fpath = os.path.join(cache_dir, model_file)
        if _is_valid_model(fpath, model_file):
            return fpath
        if os.path.isfile(fpath):
            os.remove(fpath)  # Remove corrupted cache
        url = (
            "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/"
            + path_in_repo
            + model_file
        )
        print("Downloading", model_file, "from", url)
        _robust_download(url, fpath)
        if not _is_valid_model(fpath, model_file):
            if os.path.isfile(fpath):
                os.remove(fpath)
            raise RuntimeError(f"Downloaded {model_file} is corrupted (wrong file or HTML)")
        return fpath

    emotieff_utils.download_model = _robust_download_model


@dataclass
class EmotionPrediction:
    label: str
    score: float


class EmotionRecognizer:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name or "enet_b0_8_best_vgaf"
        self.device = device
        self._model = self._build_model()

    def _build_model(self):
        _patch_emotiefflib_download()
        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
        except ImportError as exc:
            raise RuntimeError(
                "EmotiEffLib is required for emotion recognition. "
                "Install emotiefflib and update requirements."
            ) from exc
        model_name = self.model_name
        if model_name in ("default", "emotiefflib_default"):
            model_name = "enet_b0_8_best_vgaf"
        elif model_name in ("best", "enet_b2_8"):
            model_name = "enet_b2_8"  # Best in EmotiEffLib (63% AffectNet)
        elif model_name == "light":
            model_name = "mbf_va_mtl"  # 112px input, faster; torch only (no ONNX)
        # Prefer ONNX backend: avoids timm/PyTorch pickle compatibility issues
        try:
            return EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=self.device)
        except Exception:
            return EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=self.device)

    def predict_frame(self, frame: np.ndarray) -> List[EmotionPrediction]:
        # EmotiEffLib expects BGR/RGB image; predict_emotions returns (labels, scores)
        labels, scores = self._model.predict_emotions(frame, logits=False)
        predictions: List[EmotionPrediction] = []
        if isinstance(labels, str):
            labels = [labels]
            scores = scores[np.newaxis, ...]
        for i, label in enumerate(labels):
            score = float(scores[i].max()) if scores.size > 0 else 0.0
            predictions.append(EmotionPrediction(label=str(label), score=score))
        return predictions
