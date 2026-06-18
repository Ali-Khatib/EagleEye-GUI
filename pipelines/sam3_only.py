"""Mode 3: SAM 3 only — text-prompt detection (dataset class names)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.pipeline_detection import (
    class_aware_nms,
    log_pipeline_stats,
    passes_box_filters,
    remove_nested_boxes,
    run_sam3_text_detection,
    text_prompts_from_class_names,
)
from core.pipeline_config import SAM3_TEXT_CONF
from core.sam3_support import resolve_sam3_weights
from pipelines._common import PipelineResult, PipelineStats


def run(
    image_bgr: np.ndarray,
    text_classes: List[str],
    device: str,
    *,
    sam3_weights: Path | None = None,
    class_names: Optional[Dict[int, str]] = None,
) -> PipelineResult:
    t0 = time.time()
    h, w = image_bgr.shape[:2]
    weights = sam3_weights or resolve_sam3_weights()

    # Prefer explicit dataset class_names; fall back to text_classes list
    if class_names is None and text_classes:
        class_names = {i: name for i, name in enumerate(text_classes)}

    raw_dets, raw_masks = run_sam3_text_detection(
        image_bgr, class_names, device, sam3_weights=weights,
    )
    stats = PipelineStats(raw_sam3_text_count=len(raw_dets))

    filtered: List[Dict[str, Any]] = []
    filtered_masks: List[np.ndarray] = []
    for det, mask in zip(raw_dets, raw_masks):
        if passes_box_filters(det, w, h, min_conf=SAM3_TEXT_CONF):
            filtered.append(det)
            filtered_masks.append(mask)

    dets, ms, _ = remove_nested_boxes(filtered, masks=filtered_masks)
    dets, ms, _ = class_aware_nms(dets, masks=ms)

    stats.accepted_sam3_text_count = len(dets)
    stats.final_prediction_count = len(dets)
    stats.mask_count = sum(1 for m in (ms or []) if m is not None)

    log_pipeline_stats(stats)
    prompts = text_prompts_from_class_names(class_names)
    return PipelineResult(
        detections=dets,
        masks=ms or [],
        runtime_seconds=time.time() - t0,
        notes=f"SAM3-only text detect ({len(prompts)} class prompts). predicted_count={len(dets)}.",
        class_agnostic=False,
        stats=stats,
    )
