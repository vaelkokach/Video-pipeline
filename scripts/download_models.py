"""Download MMAction2 action recognition models. Supports TSM-MobileNetV2 (lightweight) and TSN-R50."""
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models" / "mmaction"

# TSM-MobileNetV2: lightweight (~2.7M params, 3.3G FLOPs) - default
TSM_MOBILENETV2_CKPT = (
    "https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/"
    "tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb/"
    "tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb_20230414-401127fd.pth"
)

# TSN-R50: heavier (~24M params) - optional fallback
TSN_R50_BASE_CONFIG = (
    "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/recognition/tsn/"
    "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py"
)
TSN_R50_CONFIG = (
    "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/recognition/tsn/"
    "tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
)
TSN_R50_CKPT = (
    "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/"
    "tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/"
    "tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth"
)


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

    # TSM-MobileNetV2 (lightweight default)
    tsm_config = MODELS_DIR / "tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py"
    tsm_ckpt = MODELS_DIR / "tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb_20230414-401127fd.pth"
    if not tsm_config.exists():
        print("TSM-MobileNetV2 config should exist at models/mmaction/ (created by repo)")
    download_file(TSM_MOBILENETV2_CKPT, tsm_ckpt)

    # Optional: TSN-R50 for higher accuracy
    base_config_dest = MODELS_DIR / "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py"
    config_dest = MODELS_DIR / "tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
    ckpt_dest = MODELS_DIR / "tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth"
    if not base_config_dest.exists():
        download_file(TSN_R50_BASE_CONFIG, base_config_dest)
    if not config_dest.exists():
        download_file(TSN_R50_CONFIG, config_dest)
    if not ckpt_dest.exists():
        download_file(TSN_R50_CKPT, ckpt_dest)


if __name__ == "__main__":
    main()
