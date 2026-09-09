"""
Resolve fine-tuned YOLO checkpoints per dataset.

VisDrone: runs/detect/train9/weights/best.pt (100-epoch run).
KITTI:    runs/detect/train2/weights/best.pt
Stock:    yolo11n.pt (COCO baseline)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from core.paths import PROJECT_ROOT, YOLO_KITTI, YOLO_STOCK, YOLO_VISDRONE


def _train_index(path: Path) -> int:
    m = re.search(r"train(\d+)", path.as_posix())
    return int(m.group(1)) if m else -1


def _collect_train_best_pts() -> List[Path]:
    found = list((PROJECT_ROOT / "runs" / "detect").glob("train*/weights/best.pt"))
    return sorted(found, key=_train_index, reverse=True)


def resolve_yolo_checkpoint(dataset: str, *, warn: bool = True) -> str:
    """Return repo-relative path to best.pt for visdrone / kitti / stock."""
    key = dataset.lower().strip()
    preferred: List[Path] = []
    if key == "visdrone":
        preferred = [YOLO_VISDRONE]
    elif key == "kitti":
        preferred = [YOLO_KITTI]
    else:
        preferred = [YOLO_STOCK]

    candidates = preferred + [p for p in _collect_train_best_pts() if p not in preferred]
    for path in candidates:
        if path.is_file():
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            if warn and path not in preferred and key in ("visdrone", "kitti"):
                print(f"[WARN] Using fallback YOLO weights: {rel}")
            elif warn and key == "visdrone" and path.resolve() == YOLO_VISDRONE.resolve():
                print(f"[INFO] VisDrone YOLO: {rel} (train9, 100-epoch best.pt)")
            return rel

    fallback = preferred[0].relative_to(PROJECT_ROOT).as_posix()
    if warn:
        print(f"[WARN] YOLO weights not found for '{key}'; expected {fallback}")
    return fallback


def yolo_checkpoint_map() -> dict[str, str]:
    """Paths for web UI / API (no console warnings)."""
    return {
        "visdrone": resolve_yolo_checkpoint("visdrone", warn=False),
        "kitti": resolve_yolo_checkpoint("kitti", warn=False),
        "stock": resolve_yolo_checkpoint("stock", warn=False),
    }
