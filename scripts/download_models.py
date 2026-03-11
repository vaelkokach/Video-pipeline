"""Download MMAction2 TSN config and checkpoint."""
import os
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models" / "mmaction"
BASE_CONFIG_URL = "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py"
CONFIG_URL = "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
CHECKPOINT_URL = "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    base_config_dest = MODELS_DIR / "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py"
    config_dest = MODELS_DIR / "tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
    ckpt_dest = MODELS_DIR / "tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth"
    download_file(BASE_CONFIG_URL, base_config_dest)
    download_file(CONFIG_URL, config_dest)
    download_file(CHECKPOINT_URL, ckpt_dest)


if __name__ == "__main__":
    main()
