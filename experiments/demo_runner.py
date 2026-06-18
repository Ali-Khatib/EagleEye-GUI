"""
Run one SAM 3 pipeline mode on the demo image and write outputs/webapp/<dataset>/<pipeline>/.

Used by exp1–exp5 scripts and the web UI.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.compare_viz import draw_detections  # noqa: E402
from core.eval_config import EVAL_IOU  # noqa: E402
from core.experiment_utils import NA, evaluate_with_ground_truth, metric_or_na  # noqa: E402
from core.sam3_support import resolve_sam3_weights  # noqa: E402
from core.yolo_weights import resolve_yolo_checkpoint  # noqa: E402
from pipelines._common import DEFAULT_TEXT_CLASSES  # noqa: E402
from pipelines import sam3_only, sam3_yolo, yolo_only, yolo_sahi, yolo_sahi_sam3  # noqa: E402

from core.webapp_outputs import (
    input_image_path,
    metrics_csv_path,
    metrics_txt_path,
    pipeline_dir,
    result_image_path,
)  # noqa: E402

OUTPUT_DIR = _root / "outputs" / "webapp"


def _resolve_image(dataset: str) -> Path:
    primary = _root / "data" / "demo" / f"test_image_{dataset}.jpg"
    fallback = _root / "data" / "demo" / "test_image.jpg"
    return primary if primary.is_file() else fallback


def _resolve_label(dataset: str) -> Path:
    primary = _root / "data" / "demo" / f"test_image_{dataset}.txt"
    fallback = _root / "data" / "demo" / "test_image.txt"
    return primary if primary.is_file() else fallback


def _file_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0) if path.is_file() else 0.0


def _draw_header(img: np.ndarray, lines: List[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(
            img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += 22


def _save_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def _save_txt(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k, v in row.items():
            f.write(f"{k}: {v}\n")


def run_demo(
    mode: str,
    *,
    mode_name: str,
    pipeline_folder: str,
    class_agnostic: bool,
    needs_sam3: bool,
    yolo_weights_override: str | None = None,
) -> None:
    dataset = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXP_DATASET", "visdrone")
    ).lower().strip()
    if dataset not in ("visdrone", "kitti", "stock"):
        print(f"[ERROR] Unknown dataset '{dataset}'.")
        sys.exit(1)

    image_path = _resolve_image(dataset)
    label_path = _resolve_label(dataset)
    if not image_path.is_file():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Could not read {image_path}")
        sys.exit(1)
    h, w = image.shape[:2]

    device = os.environ.get("EXP_DEVICE", "").strip().lower()
    if device not in ("cuda", "cpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[ERROR] CUDA/GPU requested but not available. Close other GPU jobs or install CUDA PyTorch.")
        sys.exit(1)
    print(f"[INFO] Inference device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    yolo_model: Optional[YOLO] = None
    yolo_mb = 0.0
    yolo_params = 0
    yolo_weights = ""
    if mode == "sam3_only":
        yolo_weights = resolve_yolo_checkpoint(dataset)
        yolo_path = Path(yolo_weights)
        if not yolo_path.is_file():
            yolo_path = _root / yolo_weights
        if not yolo_path.is_file():
            print(f"[ERROR] YOLO weights not found (needed for class names): {yolo_weights}")
            sys.exit(1)
        yolo_weights = str(yolo_path)
        yolo_model = YOLO(yolo_weights)
        yolo_mb = _file_mb(yolo_path)
    else:
        if yolo_weights_override:
            yolo_path = Path(yolo_weights_override)
            if not yolo_path.is_file():
                yolo_path = _root / yolo_weights_override
        else:
            yolo_weights = resolve_yolo_checkpoint(dataset)
            yolo_path = Path(yolo_weights)
            if not yolo_path.is_file():
                yolo_path = _root / yolo_weights
        if not yolo_path.is_file():
            print(f"[ERROR] YOLO weights not found: {yolo_path}")
            sys.exit(1)
        yolo_weights = str(yolo_path)
        yolo_model = YOLO(yolo_weights)
        yolo_mb = _file_mb(yolo_path)
        try:
            yolo_params = int(sum(p.numel() for p in yolo_model.model.parameters()))
        except Exception:
            yolo_params = 0

    sam3_path: Optional[Path] = None
    sam3_mb = 0.0
    if needs_sam3:
        try:
            sam3_path = resolve_sam3_weights()
            sam3_mb = _file_mb(sam3_path)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)

    t0 = time.time()
    if mode == "yolo_only":
        out = yolo_only.run(image, yolo_model, device)
    elif mode == "yolo_sahi":
        out = yolo_sahi.run(image, yolo_weights, device)
    elif mode == "sam3_only":
        out = sam3_only.run(
            image,
            list(yolo_model.names.values()) if yolo_model else DEFAULT_TEXT_CLASSES,
            device,
            sam3_weights=sam3_path,
            class_names=yolo_model.names if yolo_model else None,
        )
    elif mode == "sam3_yolo":
        out = sam3_yolo.run(
            image, yolo_model, device,
            sam3_weights=sam3_path,
            class_names=yolo_model.names if yolo_model else None,
        )
    elif mode == "yolo_sahi_sam3":
        out = yolo_sahi_sam3.run(
            image, yolo_weights, device,
            yolo_model=yolo_model,
            sam3_weights=sam3_path,
            class_names=yolo_model.names if yolo_model else None,
        )
    else:
        print(f"[ERROR] Unknown mode {mode}")
        sys.exit(1)

    runtime = time.time() - t0
    fps = 1.0 / runtime if runtime > 0 else 0.0
    detections = out.detections
    count = len(detections)
    mask_count = sum(1 for m in (out.masks or []) if m is not None)
    print(f"Mode: {mode_name} | Detections: {count} | Masks: {mask_count} | {runtime:.3f}s | FPS: {fps:.2f}")
    if out.stats:
        from core.pipeline_detection import log_pipeline_stats
        log_pipeline_stats(out.stats, prefix=mode_name)

    is_stock = dataset == "stock"
    notes = out.notes
    if is_stock:
        notes = (
            "Stock COCO weights; class-agnostic IoU vs dataset GT. " + notes
        )
        class_agnostic = True

    metric_result = evaluate_with_ground_truth(
        detections,
        str(label_path),
        w,
        h,
        str(image_path),
        EVAL_IOU,
        class_agnostic=class_agnostic,
        base_notes=notes,
    )
    for warning in metric_result["warnings"]:
        print(f"[WARN] {warning}")

    precision = metric_result["precision"]
    recall = metric_result["recall"]
    f1 = metric_result["f1"]
    accuracy = metric_result["accuracy"]
    tp_v, fp_v, fn_v = metric_result["tp"], metric_result["fp"], metric_result["fn"]
    gt_count = metric_result["ground_truth_count"]
    has_reliable_gt = metric_result["has_reliable_gt"]
    notes = metric_result["notes"]

    if has_reliable_gt:
        print(
            f"GT: {gt_count} | TP={tp_v} FP={fp_v} FN={fn_v} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

    class_names = (
        yolo_model.names
        if yolo_model is not None
        else {i: name for i, name in enumerate(DEFAULT_TEXT_CLASSES)}
    )
    vis = draw_detections(image, detections, class_names, out.masks or None)
    header = [
        f"Mode: {mode_name}",
        f"Dataset: {dataset}",
        f"Count: {count}",
        f"Runtime: {runtime:.3f}s | FPS: {fps:.2f}",
    ]
    if has_reliable_gt:
        header.append(f"P: {precision:.3f}  R: {recall:.3f}  F1: {f1:.3f}")
    else:
        header.append("P/R/F1: N/A (no ground-truth label)")
    _draw_header(vis, header)

    out_dir = pipeline_dir(_root, dataset, pipeline_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = result_image_path(out_dir)
    out_csv = metrics_csv_path(out_dir)
    out_txt = metrics_txt_path(out_dir)

    # Save copy of input so outputs/webapp/<dataset>/ shows what was run
    inp_copy = input_image_path(_root, dataset)
    inp_copy.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(inp_copy), image)

    if not cv2.imwrite(str(out_img), vis):
        print(f"[ERROR] Failed to write result image: {out_img}")
        sys.exit(1)

    summary = {
        "mode_name": mode_name,
        "dataset": dataset,
        "dataset_image": str(image_path.relative_to(_root)),
        "count": count,
        "runtime_seconds": round(runtime, 4),
        "fps": round(fps, 4),
        "yolo_model_size_mb": round(yolo_mb, 4),
        "yolo_parameter_count": yolo_params,
        "sam_checkpoint_size_mb": round(sam3_mb, 4),
        "precision": metric_or_na(precision),
        "recall": metric_or_na(recall),
        "f1_score": metric_or_na(f1),
        "accuracy": metric_or_na(accuracy),
        "iou_threshold": EVAL_IOU,
        "true_positives": metric_or_na(tp_v) if has_reliable_gt else NA,
        "false_positives": metric_or_na(fp_v) if has_reliable_gt else NA,
        "false_negatives": metric_or_na(fn_v) if has_reliable_gt else NA,
        "ground_truth_count": gt_count if has_reliable_gt else NA,
        "predicted_count": count,
        "mask_count": mask_count,
        "raw_yolo_count": getattr(out.stats, "raw_yolo_count", 0) if out.stats else 0,
        "filtered_yolo_count": getattr(out.stats, "filtered_yolo_count", 0) if out.stats else 0,
        "raw_sahi_count": getattr(out.stats, "raw_sahi_count", 0) if out.stats else 0,
        "accepted_sahi_count": getattr(out.stats, "accepted_sahi_count", 0) if out.stats else 0,
        "raw_sam3_text_count": getattr(out.stats, "raw_sam3_text_count", 0) if out.stats else 0,
        "accepted_sam3_text_count": getattr(out.stats, "accepted_sam3_text_count", 0) if out.stats else 0,
        "final_prediction_count": getattr(out.stats, "final_prediction_count", count) if out.stats else count,
        "has_reliable_gt": has_reliable_gt,
        "inference_device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "notes": notes,
    }
    _save_csv(out_csv, summary)
    _save_txt(out_txt, summary)
    print(f"Saved: {out_img}")
    print(f"Saved: {out_csv}")
