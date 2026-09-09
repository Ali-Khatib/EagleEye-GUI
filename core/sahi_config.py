"""
Shared SAHI settings — tuned for microscopic / tiny / distant objects.

Smaller tiles + high overlap + low conf = more recall (more tiles, slower).
Override via env, e.g. SAHI_SLICE=192 SAHI_OVERLAP=0.55
"""
from __future__ import annotations

import os


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    return float(raw) if raw else default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    return int(raw) if raw else default


# Tile size (px). 256 catches objects that vanish at 320+ on drone/KITTI frames.
SAHI_SLICE = _env_int("SAHI_SLICE", 256)

# Heavy overlap so nothing is lost on tile seams (microscopic objs often sit on edges).
SAHI_OVERLAP = _env_float("SAHI_OVERLAP", 0.50)

# GREEDYNMM: good for crowded slices; merge threshold tuned to keep nearby distinct objs.
SAHI_POSTPROCESS = os.environ.get("SAHI_POSTPROCESS", "GREEDYNMM")
SAHI_IOU = _env_float("SAHI_IOU", 0.40)

# Low conf — keep weak small-object activations (filter noise with SAHI_MIN_AREA).
SAHI_CONF = _env_float("SAHI_CONF", 0.12)

# Allow very small boxes (e.g. 8x8 px at 4K ≈ 64 px²).
SAHI_MIN_AREA = _env_int("SAHI_MIN_AREA", 64)

# Second-pass NMS: high IoU = only drop near-duplicate same-object merges.
SAHI_DEDUP_IOU = _env_float("SAHI_DEDUP_IOU", 0.50)

# Upscale each slice to this size before YOLO (tiny objs appear larger in 640 input).
SAHI_YOLO_IMGSZ = _env_int("SAHI_YOLO_IMGSZ", 640)
