"""Paths for EagleEye web UI single image runs (outputs/webapp/)."""
from __future__ import annotations

from pathlib import Path

PIPELINE_FOLDERS: dict[str, str] = {
    "01_yolo_only": "yolo only",
    "02_yolo_sahi": "yolo sahi",
    "03_sam3_only": "sam3 only",
    "04_sam3_yolo": "sam3 yolo",
    "05_yolo_sahi_sam3": "yolo sahi sam3",
    "06_yolov8_sahi_sam3": "yolov8 sahi sam3",
}

DATASETS = ("visdrone", "kitti", "stock")


def webapp_root(project_root: Path) -> Path:
    return project_root / "outputs" / "webapp"


def dataset_dir(project_root: Path, dataset: str) -> Path:
    return webapp_root(project_root) / dataset


def pipeline_dir(project_root: Path, dataset: str, basename_key: str) -> Path:
    folder = PIPELINE_FOLDERS.get(basename_key, basename_key)
    return dataset_dir(project_root, dataset) / folder


def input_image_path(project_root: Path, dataset: str) -> Path:
    return dataset_dir(project_root, dataset) / "input_image.jpg"


def result_image_path(pipeline_path: Path) -> Path:
    return pipeline_path / "result.jpg"


def metrics_csv_path(pipeline_path: Path) -> Path:
    return pipeline_path / "metrics.csv"


def metrics_txt_path(pipeline_path: Path) -> Path:
    return pipeline_path / "notes.txt"


def all_pipeline_dirs(project_root: Path, dataset: str) -> list[Path]:
    base = dataset_dir(project_root, dataset)
    return [base / name for name in PIPELINE_FOLDERS.values()]
