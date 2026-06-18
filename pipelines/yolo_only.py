"""Mode 1: YOLO only — full-frame detection baseline."""
from __future__ import annotations

import time
from typing import Any, List

import numpy as np

from core.detection_ops import yolo_results_to_detections
from core.eval_config import EVAL_CONF, MIN_BOX_AREA_PX, YOLO_IMGSZ
from pipelines._common import PipelineResult


def run(
    image_bgr: np.ndarray,
    yolo_model: Any,
    device: str,
) -> PipelineResult:
    t0 = time.time()
    results = yolo_model.predict(
        source=image_bgr,
        conf=EVAL_CONF,
        imgsz=YOLO_IMGSZ,
        device=device,
        verbose=False,
    )
    dets = yolo_results_to_detections(results, conf_floor=EVAL_CONF, min_area=MIN_BOX_AREA_PX)
    return PipelineResult(
        detections=dets,
        runtime_seconds=time.time() - t0,
        notes="YOLO-only baseline; classes and boxes from fine-tuned YOLO.",
        class_agnostic=False,
    )
