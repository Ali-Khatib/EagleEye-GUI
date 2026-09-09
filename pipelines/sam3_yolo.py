"""Mode 4: YOLO + SAM3 — YOLO leads; SAM3 text fills gaps; SAM3 box masks overlay only."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core.pipeline_config import SAM3_TEXT_CONF, YOLO_CONF
from core.pipeline_detection import (
    cap_sam3_text_detections,
    filter_detections_stage,
    finalize_detections,
    log_pipeline_stats,
    merge_with_priority,
    run_sam3_box_masks,
    run_sam3_text_detection,
    run_yolo_detection,
)
from core.sam3_support import resolve_sam3_weights
from pipelines._common import PipelineResult, PipelineStats


def run(
    image_bgr: np.ndarray,
    yolo_model: Any,
    device: str,
    *,
    sam3_weights: Path | None = None,
    class_names: Optional[Dict[int, str]] = None,
) -> PipelineResult:
    t0 = time.time()
    h, w = image_bgr.shape[:2]
    weights = sam3_weights or resolve_sam3_weights()
    stats = PipelineStats()

    raw_yolo = run_yolo_detection(image_bgr, yolo_model, device, class_names)
    stats.raw_yolo_count = len(raw_yolo)
    yolo_filt = filter_detections_stage(raw_yolo, w, h, YOLO_CONF)
    stats.filtered_yolo_count = len(yolo_filt)

    raw_sam3, _ = run_sam3_text_detection(image_bgr, class_names, device, sam3_weights=weights)
    stats.raw_sam3_text_count = len(raw_sam3)
    sam3_filt = filter_detections_stage(raw_sam3, w, h, SAM3_TEXT_CONF)

    merged, sam3_acc = merge_with_priority(
        yolo_filt, sam3_filt, "sam3_text",
        image_width=w, image_height=h, min_conf=SAM3_TEXT_CONF, class_names=class_names,
    )
    stats.accepted_sam3_text_count = sam3_acc
    merged = cap_sam3_text_detections(merged)

    merged, _, _ = finalize_detections(merged, w, h)
    stats.final_prediction_count = len(merged)

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
            f"YOLO+SAM3: yolo={stats.filtered_yolo_count} + sam3_text={stats.accepted_sam3_text_count} "
            f"→ final={stats.final_prediction_count}, masks={stats.mask_count}."
        ),
        class_agnostic=False,
        stats=stats,
    )
