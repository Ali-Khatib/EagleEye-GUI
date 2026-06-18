"""Shared SAHI sliced prediction (used by yolo_sahi and full stack)."""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from core.detection_ops import deduplicate_boxes, sahi_result_to_detections
from core.sahi_config import (
    SAHI_CONF,
    SAHI_DEDUP_IOU,
    SAHI_IOU,
    SAHI_MIN_AREA,
    SAHI_OVERLAP,
    SAHI_POSTPROCESS,
    SAHI_SLICE,
    SAHI_YOLO_IMGSZ,
)


def run_sahi_yolo(
    image_bgr: np.ndarray,
    yolo_weights: str,
    device: str,
    *,
    confidence_threshold: float | None = None,
) -> Tuple[List[dict], str, int]:
    """
    Run SAHI + Ultralytics YOLO on image_bgr.
    Returns (detections, notes_suffix, dedup_removed_count).
    """
    from sahi.predict import get_sliced_prediction

    from core.sahi_model_cache import get_sahi_detection_model

    conf = confidence_threshold if confidence_threshold is not None else SAHI_CONF
    sahi_model = get_sahi_detection_model(
        yolo_weights,
        device,
        confidence_threshold=conf,
        image_size=SAHI_YOLO_IMGSZ,
    )
    result = get_sliced_prediction(
        image_bgr,
        sahi_model,
        slice_height=SAHI_SLICE,
        slice_width=SAHI_SLICE,
        overlap_height_ratio=SAHI_OVERLAP,
        overlap_width_ratio=SAHI_OVERLAP,
        postprocess_type=SAHI_POSTPROCESS,
        postprocess_match_metric="IOU",
        postprocess_match_threshold=SAHI_IOU,
        verbose=0,
    )
    dets = sahi_result_to_detections(
        result, conf_floor=conf, min_area=SAHI_MIN_AREA,
    )
    dets, removed, _ = deduplicate_boxes(dets, SAHI_DEDUP_IOU, class_aware=True)
    notes = (
        f"slice={SAHI_SLICE}px overlap={SAHI_OVERLAP:.0%} "
        f"yolo_imgsz={SAHI_YOLO_IMGSZ} conf={conf} min_area={SAHI_MIN_AREA} "
        f"post={SAHI_POSTPROCESS} iou={SAHI_IOU} dedup={SAHI_DEDUP_IOU} removed={removed}"
    )
    return dets, notes, removed
