"""
Standard evaluation settings for all comparison modes.

Every pipeline uses the same confidence floor, IoU matching threshold,
and small-object definition so metrics are comparable.
"""
from __future__ import annotations

# Detection confidence (YOLO + SAHI wrapper)
EVAL_CONF = 0.20

# Greedy one-to-one matching vs ground-truth boxes
EVAL_IOU = 0.50

# GT / pred box with area < this fraction of image area counts as "small"
SMALL_OBJECT_AREA_FRAC = 0.02

# Ultralytics full-frame YOLO input size
YOLO_IMGSZ = 640

# Minimum predicted box area in pixels² (noise filter)
MIN_BOX_AREA_PX = 200

# YOLO+SAHI+SAM3 — dual-path detect (YOLO backbone + SAHI extras) + tiered SAM filter
STACK_SAHI_CONF = 0.20          # same floor as EVAL_CONF; SAHI adds slice recall on top of YOLO
STACK_MIN_SCORE = 0.20          # floor for SAHI-only survivors (= EVAL_CONF)
STACK_MIN_SCORE_PEDESTRIAN = 0.40  # SAHI-only person-like classes
STACK_MIN_SAM_MASK_FILL = 0.12  # SAM mask fill inside YOLO box (SAHI extras only)
STACK_SAM3_TEXT_CONF = 0.25     # SAM3 open-vocab text detection confidence
STACK_DEDUP_IOU = 0.45
STACK_MERGE_IOU = 0.45          # merge YOLO / SAHI / SAM3 text proposals
STACK_CONTAIN_FRAC = 0.72       # drop box if this fraction of its area lies inside a larger kept box
