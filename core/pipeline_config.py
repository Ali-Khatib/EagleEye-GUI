"""
Pipeline detection thresholds — YOLO / SAHI / SAM3 text / merge / NMS.

Used by pipelines 3, 4, 5 and shared cleanup helpers.
Override SAHI tile settings via env, e.g. SAHI_SLICE=640 SAHI_OVERLAP=0.10
"""
from __future__ import annotations

import os


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    return float(raw) if raw else default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    return int(raw) if raw else default


# Confidence floors per detector stage
YOLO_CONF = 0.35
SAHI_CONF = 0.40
SAM3_TEXT_CONF = 0.45

# Box geometry filters
MIN_BOX_AREA = 120
MAX_BOX_AREA_RATIO = 0.25
MIN_ASPECT_RATIO = 0.20
MAX_ASPECT_RATIO = 5.00

# Merge / dedup
MERGE_IOU_THRESHOLD = 0.30
FINAL_NMS_IOU = 0.50
CONTAINMENT_THRESHOLD = 0.85

# SAM3 text caps
MAX_SAM3_TEXT_DETECTIONS_PER_CLASS = 25
MAX_TOTAL_SAM3_TEXT_DETECTIONS = 80
MAX_BOX_MASKS = 60

# SAHI slice settings (pipeline 5)
SAHI_SLICE = _env_int("SAHI_SLICE", 512)
SAHI_OVERLAP = _env_float("SAHI_OVERLAP", 0.15)
SAHI_POSTPROCESS = "NMS"
SAHI_IOU = 0.60
SAHI_YOLO_IMGSZ = 640

# SAM3 box mask validation (SAHI-only suspicious drop)
SAHI_MASK_MIN_FILL = 0.08
SAHI_LOW_CONF_FOR_MASK_DROP = 0.50

# YOLO full-frame input
YOLO_IMGSZ = 640

# Source priority (higher wins on overlap)
SOURCE_PRIORITY = {"yolo": 3, "sahi": 2, "sam3_text": 1}

SAM3_TEXT_NOT_IMPLEMENTED_MSG = (
    "SAM3 text detection API is not implemented. Connect SAM3 text-prompt "
    "detection before evaluating Pipeline 3, 4, or 5."
)
