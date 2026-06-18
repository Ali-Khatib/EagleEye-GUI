"""Visualization for compare_all_modes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    class_names: Optional[Dict[int, str]] = None,
    masks: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    out = image_bgr.copy()
    if masks:
        overlay = out.copy()
        for mask in masks:
            if mask is None or mask.size == 0:
                continue
            color = (0, 200, 255)
            overlay[mask > 0] = (
                0.45 * overlay[mask > 0] + 0.55 * np.array(color, dtype=np.float32)
            ).astype(np.uint8)
        out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cid = int(det.get("class_id", -1))
        name = (class_names or {}).get(cid, f"cls{cid}")
        score = float(det.get("score", 0.0))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, f"{name} {score:.2f}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return out
