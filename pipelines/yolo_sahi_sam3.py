"""Mode 5: YOLO + SAHI + SAM3 — main research pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core.pipeline_config import SAHI_CONF, SAM3_TEXT_CONF, YOLO_CONF
from core.pipeline_detection import (
    cap_sam3_text_detections,
    class_aware_nms,
    filter_detections_stage,
    finalize_detections,
    log_pipeline_stats,
    merge_with_priority,
    remove_nested_boxes,
    run_sam3_box_masks,
    run_sam3_text_detection,
    run_sahi_yolo_detection,
    run_yolo_detection,
)
from core.sam3_support import resolve_sam3_weights
from pipelines._common import PipelineResult, PipelineStats


def run(
    image_bgr: np.ndarray,
    yolo_weights: str,
    device: str,
    *,
    yolo_model: Any = None,
    sam3_weights: Path | None = None,
    class_names: Optional[Dict[int, str]] = None,
) -> PipelineResult:
    t0 = time.time()
    if yolo_model is None:
        from ultralytics import YOLO

        yolo_model = YOLO(yolo_weights)

    h, w = image_bgr.shape[:2]
    weights = sam3_weights or resolve_sam3_weights()
    stats = PipelineStats()

    # Step 1: YOLO full-frame
    raw_yolo = run_yolo_detection(image_bgr, yolo_model, device, class_names)
    stats.raw_yolo_count = len(raw_yolo)
    yolo_filt = filter_detections_stage(raw_yolo, w, h, YOLO_CONF)
    stats.filtered_yolo_count = len(yolo_filt)

    # Step 2: SAHI sliced YOLO
    raw_sahi = run_sahi_yolo_detection(image_bgr, yolo_weights, device, class_names)
    stats.raw_sahi_count = len(raw_sahi)
    sahi_filt = filter_detections_stage(raw_sahi, w, h, SAHI_CONF)
    sahi_filt, _, _ = remove_nested_boxes(sahi_filt)
    sahi_filt, _, _ = class_aware_nms(sahi_filt)

    # Step 3: Merge YOLO + SAHI (YOLO priority)
    merged, sahi_acc = merge_with_priority(
        yolo_filt, sahi_filt, "sahi",
        image_width=w, image_height=h, min_conf=SAHI_CONF, class_names=class_names,
    )
    stats.accepted_sahi_count = sahi_acc

    # Step 4: SAM3 text detection (fills missing objects only)
    raw_sam3, _ = run_sam3_text_detection(image_bgr, class_names, device, sam3_weights=weights)
    stats.raw_sam3_text_count = len(raw_sam3)
    sam3_filt = filter_detections_stage(raw_sam3, w, h, SAM3_TEXT_CONF)
    sam3_filt, _, _ = remove_nested_boxes(sam3_filt)
    sam3_filt, _, _ = class_aware_nms(sam3_filt)

    merged, sam3_acc = merge_with_priority(
        merged, sam3_filt, "sam3_text",
        image_width=w, image_height=h, min_conf=SAM3_TEXT_CONF, class_names=class_names,
    )
    stats.accepted_sam3_text_count = sam3_acc
    merged = cap_sam3_text_detections(merged)

    # Step 6: Final cleanup before masks
    merged, _, _ = finalize_detections(merged, w, h)
    stats.final_prediction_count = len(merged)

    # Step 5: SAM3 box masks (overlay only — after merge/NMS)
    merged, masks, stats.mask_count = run_sam3_box_masks(
        image_bgr, merged, device, sam3_weights=weights,
    )
    stats.final_prediction_count = len(merged)

    log_pipeline_stats(stats)
    return PipelineResult(
        detections=merged,
        masks=masks,
        runtime_seconds=time.time() - t0,
        notes=(
            f"YOLO+SAHI+SAM3: yolo={stats.filtered_yolo_count} sahi+={stats.accepted_sahi_count} "
            f"sam3_text+={stats.accepted_sam3_text_count} final={stats.final_prediction_count} "
            f"masks={stats.mask_count}."
        ),
        class_agnostic=False,
        stats=stats,
    )
