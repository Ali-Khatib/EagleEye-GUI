"""Reuse SAHI AutoDetectionModel across images (avoid reloading YOLO weights each image)."""
from __future__ import annotations

from typing import Any, Dict


_sahi_models: Dict[str, Any] = {}


def get_sahi_detection_model(
    yolo_weights: str,
    device: str,
    *,
    confidence_threshold: float,
    image_size: int,
) -> Any:
    from sahi import AutoDetectionModel

    key = f"{yolo_weights}|{device}|{confidence_threshold}|{image_size}"
    if key not in _sahi_models:
        _sahi_models[key] = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=yolo_weights,
            confidence_threshold=confidence_threshold,
            device=device,
            image_size=image_size,
        )
    return _sahi_models[key]


def preload_sahi_detection_model(
    yolo_weights: str,
    device: str,
    *,
    confidence_threshold: float,
    image_size: int,
) -> None:
    get_sahi_detection_model(
        yolo_weights,
        device,
        confidence_threshold=confidence_threshold,
        image_size=image_size,
    )
