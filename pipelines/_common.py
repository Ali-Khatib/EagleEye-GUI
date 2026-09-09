"""Shared types and helpers for pipeline modes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PipelineStats:
    raw_yolo_count: int = 0
    filtered_yolo_count: int = 0
    raw_sahi_count: int = 0
    accepted_sahi_count: int = 0
    raw_sam3_text_count: int = 0
    accepted_sam3_text_count: int = 0
    mask_count: int = 0
    final_prediction_count: int = 0


@dataclass
class PipelineResult:
    detections: List[Dict[str, Any]]
    masks: List[np.ndarray] = field(default_factory=list)
    runtime_seconds: float = 0.0
    notes: str = ""
    class_agnostic: bool = False
    stats: Optional[PipelineStats] = None


# Broad street/traffic list for sam3_only (open-vocab, class-agnostic eval).
# Pipelines 4 & 5 use YOLO class names as SAM3 text prompts (car, pedestrian, …)
# plus YOLO/SAHI box prompts for mask refinement.
DEFAULT_TEXT_CLASSES = [
    # People & riders
    "person",
    "pedestrian",
    "people",
    "crowd",
    "cyclist",
    "bicycle",
    "motorcycle",
    "motorcyclist",
    "motor",
    "scooter",
    "skateboard",
    "rider",
    "wheelchair",
    "stroller",
    # Road vehicles
    "car",
    "van",
    "truck",
    "bus",
    "trailer",
    "tram",
    "train",
    "taxi",
    "suv",
    "pickup truck",
    "tricycle",
    "awning-tricycle",
    "construction vehicle",
    "excavator",
    "forklift",
    "ambulance",
    "police car",
    "garbage truck",
    # Aviation (VisDrone / street-adjacent)
    "drone",
    "airplane",
    "helicopter",
    # Infrastructure & furniture
    "traffic light",
    "traffic sign",
    "stop sign",
    "street light",
    "pole",
    "lamp post",
    "fire hydrant",
    "mailbox",
    "bench",
    "fence",
    "barrier",
    "traffic cone",
    "cone",
    "trash can",
    "garbage bin",
    # Animals
    "dog",
    "cat",
    "bird",
    "animal",
    # Carry / misc
    "umbrella",
    "handbag",
    "backpack",
    "suitcase",
    "boat",
    "ship",
    "cart",
    "stand",
    "table",
    "chair",
]
