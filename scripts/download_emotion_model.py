"""Pre-download EmotiEffLib emotion model to avoid 429 rate-limit during pipeline run.
Downloads ONNX model by default (no timm dependency); use --torch for PyTorch model."""
from __future__ import annotations

import argparse
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference_emotion import _patch_emotiefflib_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", action="store_true", help="Download PyTorch model (requires timm)")
    args = parser.parse_args()
    _patch_emotiefflib_download()
    model_name = "enet_b0_8_best_vgaf"
    if args.torch:
        from emotiefflib.utils import get_model_path_torch
        print(f"Downloading PyTorch model ({model_name})...")
        path = get_model_path_torch(model_name)
    else:
        from emotiefflib.utils import get_model_path_onnx
        print(f"Downloading ONNX model ({model_name})...")
        path = get_model_path_onnx(model_name)
    print(f"Ok: {path}")


if __name__ == "__main__":
    main()
