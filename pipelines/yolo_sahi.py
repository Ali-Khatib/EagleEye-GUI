"""Mode 2: YOLO + SAHI — sliced inference for microscopic / crowded objects."""

from __future__ import annotations



import time

from typing import List



import numpy as np



from core.sahi_runner import run_sahi_yolo

from pipelines._common import PipelineResult





def run(

    image_bgr: np.ndarray,

    yolo_weights: str,

    device: str,

) -> PipelineResult:

    t0 = time.time()

    dets, notes, _ = run_sahi_yolo(image_bgr, yolo_weights, device)

    return PipelineResult(

        detections=dets,

        runtime_seconds=time.time() - t0,

        notes=f"YOLO+SAHI (microscopic-tuned): {notes}.",

        class_agnostic=False,

    )

