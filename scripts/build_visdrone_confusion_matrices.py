#!/usr/bin/env python3
"""Build class confusion matrices for VisDrone test-dev (YOLO11 + YOLOv8 pipelines).

By default reads saved prediction labels from benchmark output (no GPU inference).
Use --run-inference only if you explicitly want to re-run all pipelines on 1610 images.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.eval_config import EVAL_IOU  # noqa: E402
from core.experiment_utils import (  # noqa: E402
    calculate_iou,
    load_ground_truth_yolo_labels_checked,
    load_prediction_yolo_labels,
)
from core.paths import YOLO_VISDRONE, YOLO_VISDRONE_V8  # noqa: E402
from pipelines.compare_all import discover_pairs  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402

VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

NUM_CLASSES = len(VISDRONE_CLASSES)
SKLEARN_LABELS = list(range(NUM_CLASSES + 2))
Y_TRUE_FALSE_ALARM = NUM_CLASSES
Y_PRED_MISSED = NUM_CLASSES + 1

PIPELINE_LABELS = {
    "yolo_only": "YOLO only",
    "yolo_sahi": "YOLO + SAHI",
    "sam3_only": "SAM3 only",
    "sam3_yolo": "SAM3 + YOLO",
    "yolo_sahi_sam3": "YOLO + SAHI + SAM3",
}

MATRIX_JOBS: List[Tuple[str, str, str, Path, Path]] = [
    (
        "1", "YOLO11", "yolo_only", YOLO_VISDRONE,
        ROOT / "benchmark_working/supervisor_report/visdrone_test_dev/yolo_only",
    ),
    (
        "2", "YOLO11", "yolo_sahi", YOLO_VISDRONE,
        ROOT / "benchmark_working/supervisor_report/visdrone_test_dev/yolo_sahi",
    ),
    (
        "3", "YOLO11", "sam3_only", YOLO_VISDRONE,
        ROOT / "benchmark_working/supervisor_report/visdrone_test_dev/sam3_only",
    ),
    (
        "4", "YOLO11", "sam3_yolo", YOLO_VISDRONE,
        ROOT / "benchmark_working/supervisor_report/visdrone_test_dev/sam3_yolo",
    ),
    (
        "5", "YOLO11", "yolo_sahi_sam3", YOLO_VISDRONE,
        ROOT / "benchmark_working/supervisor_report/visdrone_test_dev/yolo_sahi_sam3",
    ),
    (
        "6", "YOLOv8", "yolo_sahi_sam3", YOLO_VISDRONE_V8,
        ROOT / "benchmark_working/supervisor_report/visdrone_test_dev_yolov8/yolo_sahi_sam3",
    ),
]

MODE_CLASS_AGNOSTIC = {mode: False for mode in PIPELINE_LABELS}


def job_filename(number: str, backbone: str, mode: str) -> str:
    pipeline = PIPELINE_LABELS[mode]
    if backbone == "YOLOv8":
        return f"{number} - YOLOv8 - {pipeline}"
    if mode == "sam3_only":
        return f"{number} - {pipeline}"
    return f"{number} - YOLO11 - {pipeline}"


def job_title(number: str, backbone: str, mode: str) -> str:
    return job_filename(number, backbone, mode)


def _clip_class_id(class_id: int) -> int:
    if class_id < 0 or class_id >= NUM_CLASSES:
        return min(max(int(class_id), 0), NUM_CLASSES - 1)
    return int(class_id)


def collect_image_pairs(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_thresh: float,
    class_agnostic: bool,
) -> Tuple[List[int], List[int]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    used_gt = [False] * len(ground_truth)
    preds_sorted = sorted(predictions, key=lambda d: d.get("score", 0.0), reverse=True)

    for pred in preds_sorted:
        pred_cls = _clip_class_id(int(pred.get("class_id", 0)))
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(ground_truth):
            if used_gt[idx]:
                continue
            if not class_agnostic and pred_cls != _clip_class_id(int(gt.get("class_id", 0))):
                continue
            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0 and best_iou >= iou_thresh:
            gt_cls = _clip_class_id(int(ground_truth[best_idx].get("class_id", 0)))
            y_true.append(gt_cls)
            y_pred.append(pred_cls)
            used_gt[best_idx] = True
        else:
            y_true.append(Y_TRUE_FALSE_ALARM)
            y_pred.append(pred_cls)

    for idx, gt in enumerate(ground_truth):
        if not used_gt[idx]:
            y_true.append(_clip_class_id(int(gt.get("class_id", 0))))
            y_pred.append(Y_PRED_MISSED)

    return y_true, y_pred


def pairs_to_display_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=SKLEARN_LABELS)
    row_idx = list(range(NUM_CLASSES)) + [Y_TRUE_FALSE_ALARM]
    col_idx = list(range(NUM_CLASSES)) + [Y_PRED_MISSED]
    return cm[np.ix_(row_idx, col_idx)].astype(np.int64)


def _image_size(img_path: Path) -> Tuple[int, int] | None:
    try:
        with Image.open(img_path) as im:
            w, h = im.size
        return w, h
    except OSError:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            return None
        h, w = image_bgr.shape[:2]
        return w, h


def count_saved_labels(labels_dir: Path, pairs: List[Tuple[Path, Path]]) -> int:
    if not labels_dir.is_dir():
        return 0
    return sum(1 for img_path, _ in pairs if (labels_dir / f"{img_path.stem}.txt").is_file())


def build_matrix_from_labels(
    number: str,
    backbone: str,
    mode: str,
    benchmark_dir: Path,
    pairs: List[Tuple[Path, Path]],
) -> np.ndarray:
    labels_dir = benchmark_dir / "labels"
    have = count_saved_labels(labels_dir, pairs)
    if have == 0:
        raise FileNotFoundError(
            f"No prediction labels in {labels_dir}\n"
            "The comparison_report only stores aggregate metrics (P/R/F1/mAP), not "
            "per-box class assignments. Export labels once with:\n"
            "  .\\scripts\\export_visdrone_cm_labels.ps1\n"
            "Then re-run this script (instant, no GPU)."
        )
    if have < len(pairs):
        print(f"[WARN] {job_filename(number, backbone, mode)}: {have}/{len(pairs)} label files found")

    class_agnostic = MODE_CLASS_AGNOSTIC[mode]
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    desc = job_filename(number, backbone, mode)

    for img_path, lbl_path in tqdm(pairs, desc=desc, unit="img"):
        pred_path = labels_dir / f"{img_path.stem}.txt"
        if not pred_path.is_file():
            continue
        size = _image_size(img_path)
        if size is None:
            continue
        w, h = size
        gt_r = load_ground_truth_yolo_labels_checked(
            str(lbl_path), w, h, image_path=str(img_path),
        )
        if not gt_r["reliable"]:
            continue
        preds = load_prediction_yolo_labels(str(pred_path), w, h)
        yt, yp = collect_image_pairs(preds, gt_r["boxes"], EVAL_IOU, class_agnostic)
        y_true_all.extend(yt)
        y_pred_all.extend(yp)

    if not y_true_all:
        raise RuntimeError(f"No matched detections for {desc}; check {labels_dir}")

    return pairs_to_display_matrix(y_true_all, y_pred_all)


def build_matrix_from_inference(
    number: str,
    backbone: str,
    mode: str,
    weights: Path,
    benchmark_dir: Path,
    pairs: List[Tuple[Path, Path]],
    device: str,
    sam3_path: Path | None,
    save_labels: bool = True,
) -> np.ndarray:
    import torch
    from ultralytics import YOLO

    from core.detection_ops import detections_to_yolo_txt  # noqa: E402
    from core.pipeline_config import SAHI_CONF, SAHI_YOLO_IMGSZ  # noqa: E402
    from core.sahi_model_cache import preload_sahi_detection_model  # noqa: E402
    from core.sam3_support import resolve_sam3_weights  # noqa: E402
    from pipelines import sam3_only, sam3_yolo, yolo_only, yolo_sahi, yolo_sahi_sam3  # noqa: E402

    if sam3_path is None and mode in ("sam3_only", "sam3_yolo", "yolo_sahi_sam3"):
        sam3_path = resolve_sam3_weights()

    class_agnostic = MODE_CLASS_AGNOSTIC[mode]
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    yolo_weights = str(weights)
    yolo_model = YOLO(yolo_weights)
    class_names = yolo_model.names
    if mode in ("yolo_sahi", "yolo_sahi_sam3"):
        preload_sahi_detection_model(
            yolo_weights,
            device,
            confidence_threshold=SAHI_CONF,
            image_size=SAHI_YOLO_IMGSZ,
        )

    desc = job_filename(number, backbone, mode)
    labels_dir = benchmark_dir / "labels"
    if save_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in tqdm(pairs, desc=desc, unit="img"):
        pred_path = labels_dir / f"{img_path.stem}.txt"
        cached = save_labels and pred_path.is_file()

        if cached:
            size = _image_size(img_path)
            if size is None:
                continue
            w, h = size
        else:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue
            h, w = image_bgr.shape[:2]

        gt_r = load_ground_truth_yolo_labels_checked(
            str(lbl_path), w, h, image_path=str(img_path),
        )
        if not gt_r["reliable"]:
            continue

        if cached:
            detections = load_prediction_yolo_labels(str(pred_path), w, h)
        elif mode == "yolo_only":
            out = yolo_only.run(image_bgr, yolo_model, device)
            detections = out.detections
        elif mode == "yolo_sahi":
            out = yolo_sahi.run(image_bgr, yolo_weights, device)
            detections = out.detections
        elif mode == "sam3_only":
            out = sam3_only.run(
                image_bgr, list(class_names.values()), device,
                sam3_weights=sam3_path, class_names=class_names,
            )
            detections = out.detections
        elif mode == "sam3_yolo":
            out = sam3_yolo.run(
                image_bgr, yolo_model, device,
                sam3_weights=sam3_path, class_names=class_names,
            )
            detections = out.detections
        elif mode == "yolo_sahi_sam3":
            out = yolo_sahi_sam3.run(
                image_bgr, yolo_weights, device,
                yolo_model=yolo_model, sam3_weights=sam3_path, class_names=class_names,
            )
            detections = out.detections
        else:
            raise ValueError(mode)

        if save_labels and not cached:
            pred_path.write_text(
                detections_to_yolo_txt(detections, w, h), encoding="utf-8",
            )

        yt, yp = collect_image_pairs(detections, gt_r["boxes"], EVAL_IOU, class_agnostic)
        y_true_all.extend(yt)
        y_pred_all.extend(yp)

    return pairs_to_display_matrix(y_true_all, y_pred_all)


def save_png(path: Path, matrix: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    row_labels = VISDRONE_CLASSES + ["False alarm (no GT box)"]
    col_labels = VISDRONE_CLASSES + ["Missed (no prediction)"]
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=row_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisDrone confusion matrices from saved benchmark labels (default) or inference.",
    )
    parser.add_argument("--output", default=str(ROOT / "confusion matrices"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="", help="Comma-separated matrix numbers, e.g. 1,5,6")
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Re-run full GPU inference on all images (slow; not needed if labels are exported)",
    )
    parser.add_argument("--device", default="", help="cuda or cpu (inference mode only)")
    args = parser.parse_args()

    src = ROOT / "dataset/VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/images"
    gt = ROOT / "dataset/VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/labels"
    pairs = discover_pairs(src, gt)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    out_dir = Path(args.output)
    only_nums = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None

    device = args.device
    sam3_path = None
    if args.run_inference:
        import torch
        from core.sam3_support import resolve_sam3_weights

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam3_path = resolve_sam3_weights()

    mode_label = "inference (GPU)" if args.run_inference else "saved labels (no GPU)"
    print(f"[INFO] {len(pairs)} images | {mode_label} | output={out_dir}")

    for number, backbone, mode, weights, benchmark_dir in MATRIX_JOBS:
        if only_nums and number not in only_nums:
            continue
        stem = job_filename(number, backbone, mode)
        png_path = out_dir / f"{stem}.png"
        if png_path.is_file() and not args.only:
            print(f"[SKIP] {stem} — already exists")
            continue

        print(f"\n{'=' * 60}\n{stem}\n{'=' * 60}")
        if args.run_inference:
            matrix = build_matrix_from_inference(
                number, backbone, mode, weights, benchmark_dir, pairs, device, sam3_path,
            )
        else:
            matrix = build_matrix_from_labels(number, backbone, mode, benchmark_dir, pairs)
        save_png(png_path, matrix, job_title(number, backbone, mode))
        print(f"[DONE] {png_path}")

    print(f"\n[DONE] {out_dir}")


if __name__ == "__main__":
    main()
