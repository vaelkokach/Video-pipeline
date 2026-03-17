"""
Fix MMAction2 pip package: the DRN module is missing from the published wheel.
This patches mmaction.models.localizers to skip the DRN import (not needed for TSN).
Run after: pip install mmaction2
"""
from __future__ import annotations

import sys
from pathlib import Path


def find_mmaction_localizers() -> Path | None:
    for p in sys.path:
        init = Path(p) / "mmaction" / "models" / "localizers" / "__init__.py"
        if init.exists():
            return init
    return None


def patch_localizers_init(path: Path) -> bool:
    content = path.read_text(encoding="utf-8")
    if "from .drn.drn import DRN" not in content:
        return False  # Already patched
    new_content = content.replace(
        "from .drn.drn import DRN\n",
        "# from .drn.drn import DRN  # Missing in pip package; not needed for TSN\n",
    ).replace(
        "__all__ = ['TEM', 'PEM', 'BMN', 'TCANet', 'DRN']",
        "__all__ = ['TEM', 'PEM', 'BMN', 'TCANet']  # 'DRN' excluded",
    )
    path.write_text(new_content, encoding="utf-8")
    return True


def main() -> None:
    init_path = find_mmaction_localizers()
    if not init_path:
        print("mmaction not found. Install with: pip install mmaction2")
        sys.exit(1)
    if patch_localizers_init(init_path):
        print(f"Patched {init_path}")
    else:
        print("Already patched or no change needed.")


if __name__ == "__main__":
    main()
