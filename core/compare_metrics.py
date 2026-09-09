"""Aggregate metrics and recommendations for compare_all_modes."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.detection_ops import count_duplicate_predictions, is_small_box
from core.eval_config import EVAL_CONF, EVAL_IOU, SMALL_OBJECT_AREA_FRAC
from core.experiment_utils import evaluate_detections, load_ground_truth_yolo_labels_checked


@dataclass
class ImageMetrics:
    image: str
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    fn: int
    ground_truth_count: int
    predicted_count: int
    small_object_recall: float
    duplicate_pairs: int
    runtime_seconds: float
    mask_quality: float
    has_reliable_gt: bool = True
    raw_yolo_count: int = 0
    filtered_yolo_count: int = 0
    raw_sahi_count: int = 0
    accepted_sahi_count: int = 0
    raw_sam3_text_count: int = 0
    accepted_sam3_text_count: int = 0
    mask_count: int = 0
    final_prediction_count: int = 0


@dataclass
class ModeAggregate:
    mode: str
    images_evaluated: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    iou_threshold: float = EVAL_IOU
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_predictions: int = 0
    total_ground_truth: int = 0
    total_runtime: float = 0.0
    small_object_recall: float = 0.0
    fps: float = 0.0
    avg_inference_time: float = 0.0
    false_positives: int = 0
    missed_objects: int = 0
    duplicate_detection_count: int = 0
    mask_quality: float = 0.0
    notes: str = ""
    per_image: List[ImageMetrics] = field(default_factory=list)
    avg_raw_yolo_count: float = 0.0
    avg_filtered_yolo_count: float = 0.0
    avg_raw_sahi_count: float = 0.0
    avg_accepted_sahi_count: float = 0.0
    avg_raw_sam3_text_count: float = 0.0
    avg_accepted_sam3_text_count: float = 0.0
    avg_mask_count: float = 0.0
    avg_final_prediction_count: float = 0.0


def _small_object_recall(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    iou_thresh: float,
    class_agnostic: bool,
) -> float:
    small_gt = [
        g for g in ground_truth
        if is_small_box(g["bbox"], img_w, img_h, SMALL_OBJECT_AREA_FRAC)
    ]
    if not small_gt:
        return 1.0
    used = [False] * len(small_gt)
    preds = sorted(predictions, key=lambda d: d.get("score", 0.0), reverse=True)
    tp = 0
    for pred in preds:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(small_gt):
            if used[j]:
                continue
            if not class_agnostic and pred.get("class_id") != gt.get("class_id"):
                continue
            from core.experiment_utils import calculate_iou
            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            tp += 1
            used[best_j] = True
    return tp / len(small_gt)


def compute_map50_pooled(
    all_preds: List[Dict[str, Any]],
    all_gt: List[Dict[str, Any]],
    class_agnostic: bool = False,
) -> float:
    """
    Macro-averaged AP@0.5 across classes (pooled over all images).
    Simplified but consistent across modes on the same GT.
    """
    if not all_gt:
        return 0.0
    if class_agnostic:
        classes = [0]
        preds_by_c = {0: all_preds}
        gt_by_c = {0: all_gt}
    else:
        classes = sorted({int(g["class_id"]) for g in all_gt})
        preds_by_c = {c: [p for p in all_preds if int(p.get("class_id", -1)) == c] for c in classes}
        gt_by_c = {c: [g for g in all_gt if int(g["class_id"]) == c] for c in classes}

    aps: List[float] = []
    for c in classes:
        preds = sorted(preds_by_c.get(c, []), key=lambda p: p.get("score", 0.0), reverse=True)
        gt = gt_by_c.get(c, [])
        if not gt:
            continue
        matched = [False] * len(gt)
        tp_cum: List[int] = []
        fp_cum: List[int] = []
        tp, fp = 0, 0
        for pred in preds:
            from core.experiment_utils import calculate_iou
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt):
                if matched[j]:
                    continue
                iou = calculate_iou(pred["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= EVAL_IOU:
                tp += 1
                matched[best_j] = True
            else:
                fp += 1
            tp_cum.append(tp)
            fp_cum.append(fp)
        if not tp_cum:
            aps.append(0.0)
            continue
        recalls = [tp_cum[i] / len(gt) for i in range(len(tp_cum))]
        precisions = [tp_cum[i] / (tp_cum[i] + fp_cum[i]) if (tp_cum[i] + fp_cum[i]) else 0.0
                      for i in range(len(tp_cum))]
        ap = 0.0
        for t in [i / 10.0 for i in range(11)]:
            p_at_r = max((precisions[i] for i, r in enumerate(recalls) if r >= t), default=0.0)
            ap += p_at_r / 11.0
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0


def compute_map_at_iou(
    all_preds: List[Dict[str, Any]],
    all_gt: List[Dict[str, Any]],
    iou_thresh: float,
    class_agnostic: bool = False,
) -> float:
    if not all_gt:
        return 0.0
    if class_agnostic:
        classes = [0]
        preds_by_c = {0: all_preds}
        gt_by_c = {0: all_gt}
    else:
        classes = sorted({int(g["class_id"]) for g in all_gt})
        preds_by_c = {c: [p for p in all_preds if int(p.get("class_id", -1)) == c] for c in classes}
        gt_by_c = {c: [g for g in all_gt if int(g["class_id"]) == c] for c in classes}

    aps: List[float] = []
    for c in classes:
        preds = sorted(preds_by_c.get(c, []), key=lambda p: p.get("score", 0.0), reverse=True)
        gt = gt_by_c.get(c, [])
        if not gt:
            continue
        matched = [False] * len(gt)
        tp_cum: List[int] = []
        fp_cum: List[int] = []
        tp, fp = 0, 0
        for pred in preds:
            from core.experiment_utils import calculate_iou
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt):
                if matched[j]:
                    continue
                iou = calculate_iou(pred["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_thresh:
                tp += 1
                matched[best_j] = True
            else:
                fp += 1
            tp_cum.append(tp)
            fp_cum.append(fp)
        if not tp_cum:
            aps.append(0.0)
            continue
        recalls = [tp_cum[i] / len(gt) for i in range(len(tp_cum))]
        precisions = [
            tp_cum[i] / (tp_cum[i] + fp_cum[i]) if (tp_cum[i] + fp_cum[i]) else 0.0
            for i in range(len(tp_cum))
        ]
        ap = 0.0
        for t in [i / 10.0 for i in range(11)]:
            p_at_r = max((precisions[i] for i, r in enumerate(recalls) if r >= t), default=0.0)
            ap += p_at_r / 11.0
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0


def compute_map50_95_pooled(
    all_preds: List[Dict[str, Any]],
    all_gt: List[Dict[str, Any]],
    class_agnostic: bool = False,
) -> float:
    if not all_gt:
        return 0.0
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    return sum(
        compute_map_at_iou(all_preds, all_gt, t, class_agnostic=class_agnostic)
        for t in thresholds
    ) / len(thresholds)


def mean_mask_confidence(predictions: List[Dict[str, Any]], masks: List[Any]) -> float:
    """Average detection confidence when masks are present (proxy for mask quality)."""
    if not predictions:
        return 0.0
    if masks and len(masks) == len(predictions):
        scores = [float(d.get("score", 0.0)) for d in predictions]
        return sum(scores) / len(scores) if scores else 0.0
    scores = [float(d.get("score", 0.0)) for d in predictions]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_image(
    image_path: Path,
    label_path: Path,
    predictions: List[Dict[str, Any]],
    runtime_seconds: float,
    class_agnostic: bool,
    masks: Optional[List[Any]] = None,
    pipeline_stats: Optional[Any] = None,
    image_hw: Optional[tuple[int, int]] = None,
) -> ImageMetrics:
    if image_hw is not None:
        h, w = image_hw
    else:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        h, w = img.shape[:2]
    gt_result = load_ground_truth_yolo_labels_checked(
        str(label_path), w, h, image_path=str(image_path),
    )
    gt = gt_result["boxes"]
    if not gt_result["reliable"]:
        return ImageMetrics(
            image=image_path.name,
            precision=0.0, recall=0.0, f1=0.0, accuracy=0.0,
            tp=0, fp=0, fn=len(gt),
            ground_truth_count=len(gt),
            predicted_count=len(predictions),
            small_object_recall=0.0,
            duplicate_pairs=count_duplicate_predictions(predictions),
            runtime_seconds=runtime_seconds,
            mask_quality=mean_mask_confidence(predictions, masks or []),
            has_reliable_gt=False,
        )
    m = evaluate_detections(predictions, gt, EVAL_IOU, class_agnostic)
    sor = _small_object_recall(predictions, gt, w, h, EVAL_IOU, class_agnostic)
    ps = pipeline_stats
    return ImageMetrics(
        image=image_path.name,
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        accuracy=m["accuracy"],
        tp=m["tp"],
        fp=m["fp"],
        fn=m["fn"],
        ground_truth_count=len(gt),
        predicted_count=len(predictions),
        small_object_recall=sor,
        duplicate_pairs=count_duplicate_predictions(predictions),
        runtime_seconds=runtime_seconds,
        mask_quality=mean_mask_confidence(predictions, masks or []),
        raw_yolo_count=getattr(ps, "raw_yolo_count", 0),
        filtered_yolo_count=getattr(ps, "filtered_yolo_count", 0),
        raw_sahi_count=getattr(ps, "raw_sahi_count", 0),
        accepted_sahi_count=getattr(ps, "accepted_sahi_count", 0),
        raw_sam3_text_count=getattr(ps, "raw_sam3_text_count", 0),
        accepted_sam3_text_count=getattr(ps, "accepted_sam3_text_count", 0),
        mask_count=getattr(ps, "mask_count", 0),
        final_prediction_count=getattr(ps, "final_prediction_count", len(predictions)),
    )


def aggregate_mode(mode: str, rows: List[ImageMetrics], notes: str) -> ModeAggregate:
    valid = [r for r in rows if r.has_reliable_gt]
    if not valid:
        return ModeAggregate(mode=mode, notes=notes or "No reliable GT labels.")
    n = len(valid)
    total_time = sum(r.runtime_seconds for r in valid)
    total_tp = sum(r.tp for r in valid)
    total_fp = sum(r.fp for r in valid)
    total_fn = sum(r.fn for r in valid)
    total_pred = sum(r.predicted_count for r in valid)
    total_gt = sum(r.ground_truth_count for r in valid)
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) else 0.0
    return ModeAggregate(
        mode=mode,
        images_evaluated=n,
        precision=prec,
        recall=rec,
        f1=f1,
        accuracy=acc,
        small_object_recall=sum(r.small_object_recall for r in valid) / n,
        fps=(n / total_time) if total_time > 0 else 0.0,
        avg_inference_time=total_time / n,
        total_runtime=total_time,
        false_positives=total_fp,
        missed_objects=total_fn,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_predictions=total_pred,
        total_ground_truth=total_gt,
        duplicate_detection_count=sum(r.duplicate_pairs for r in valid),
        mask_quality=sum(r.mask_quality for r in valid) / n,
        iou_threshold=EVAL_IOU,
        notes=notes,
        per_image=rows,
        avg_raw_yolo_count=sum(r.raw_yolo_count for r in valid) / n,
        avg_filtered_yolo_count=sum(r.filtered_yolo_count for r in valid) / n,
        avg_raw_sahi_count=sum(r.raw_sahi_count for r in valid) / n,
        avg_accepted_sahi_count=sum(r.accepted_sahi_count for r in valid) / n,
        avg_raw_sam3_text_count=sum(r.raw_sam3_text_count for r in valid) / n,
        avg_accepted_sam3_text_count=sum(r.accepted_sam3_text_count for r in valid) / n,
        avg_mask_count=sum(r.mask_count for r in valid) / n,
        avg_final_prediction_count=sum(r.final_prediction_count for r in valid) / n,
    )


def finalize_mode_maps(
    agg: ModeAggregate,
    all_preds: List[Dict[str, Any]],
    all_gt: List[Dict[str, Any]],
    class_agnostic: bool,
) -> ModeAggregate:
    agg.map50 = compute_map50_pooled(all_preds, all_gt, class_agnostic=class_agnostic)
    agg.map50_95 = compute_map50_95_pooled(all_preds, all_gt, class_agnostic=class_agnostic)
    return agg


def recommend_mode(aggregates: Dict[str, ModeAggregate]) -> Dict[str, str]:
    """Rule-based recommendation from measured metrics (not hardcoded winners)."""
    valid = {k: v for k, v in aggregates.items() if v.images_evaluated > 0}
    if not valid:
        return {"recommendation": "No valid evaluations.", "best_overall": "", "speed": ""}

    speed_best = max(valid.items(), key=lambda kv: kv[1].fps)[0]
    f1_best = max(valid.items(), key=lambda kv: kv[1].f1)[0]
    small_best = max(valid.items(), key=lambda kv: kv[1].small_object_recall)[0]
    mask_best = max(valid.items(), key=lambda kv: kv[1].mask_quality)[0]

    full = valid.get("yolo_sahi_sam3")
    yolo_only = valid.get("yolo_only")
    sam3_only = valid.get("sam3_only")

    lines: List[str] = []
    if yolo_only:
        lines.append(
            f"Speed: {speed_best} at {valid[speed_best].fps:.2f} FPS — "
            f"yolo_only is fastest ({valid['yolo_only'].fps:.2f} FPS) but less complete on small/crowded objects."
        )
    if full:
        lines.append(
            f"Main pipeline yolo_sahi_sam3: F1={full.f1:.3f}, small-object recall={full.small_object_recall:.3f}, "
            f"mask_quality={full.mask_quality:.3f}."
        )
        if full.small_object_recall >= valid[small_best].small_object_recall * 0.98 and full.mask_quality >= valid[mask_best].mask_quality * 0.95:
            lines.append("Best overall for this project: yolo_sahi_sam3 (small objects + mask refinement).")
    if sam3_only:
        lines.append(
            f"sam3_only: powerful text segmentation but box F1={sam3_only.f1:.3f} vs YOLO-guided modes — "
            "less controlled than detector-first pipelines for strict box mAP."
        )
    lines.append(f"Highest box F1: {f1_best} ({valid[f1_best].f1:.3f}).")

    best_overall = "yolo_sahi_sam3"
    if full and full.small_object_recall >= max(v.small_object_recall for v in valid.values()) * 0.98:
        best_overall = "yolo_sahi_sam3"
    elif f1_best:
        best_overall = f1_best

    return {
        "recommendation": " ".join(lines),
        "best_overall": best_overall,
        "speed": speed_best,
        "small_objects": small_best,
        "mask_quality": mask_best,
        "box_f1": f1_best,
    }


def load_mode_aggregate_from_metrics(mode_dir: Path) -> ModeAggregate | None:
    """Rebuild ModeAggregate from metrics.json (for --skip-existing resume)."""
    p = mode_dir / "metrics.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return ModeAggregate(
        mode=str(d.get("mode", mode_dir.name)),
        images_evaluated=int(d.get("images_evaluated", 0)),
        precision=float(d.get("precision", 0.0)),
        recall=float(d.get("recall", 0.0)),
        f1=float(d.get("f1", 0.0)),
        accuracy=float(d.get("accuracy", 0.0)),
        map50=float(d.get("map50", 0.0)),
        map50_95=float(d.get("map50_95", 0.0)),
        iou_threshold=float(d.get("iou_threshold", EVAL_IOU)),
        total_tp=int(d.get("true_positives", 0)),
        total_fp=int(d.get("false_positives", 0)),
        total_fn=int(d.get("false_negatives", 0)),
        total_predictions=int(d.get("total_predictions", 0)),
        total_ground_truth=int(d.get("total_ground_truth", 0)),
        total_runtime=float(d.get("total_runtime_seconds", 0.0)),
        small_object_recall=float(d.get("small_object_recall", 0.0)),
        fps=float(d.get("fps", 0.0)),
        avg_inference_time=float(d.get("avg_inference_time", 0.0)),
        false_positives=int(d.get("false_positives", 0)),
        missed_objects=int(d.get("false_negatives", 0)),
        duplicate_detection_count=int(d.get("duplicate_detection_count", 0)),
        mask_quality=float(d.get("mask_quality", 0.0)),
        notes=str(d.get("notes", "")),
        avg_raw_yolo_count=float(d.get("avg_raw_yolo_count", 0.0)),
        avg_filtered_yolo_count=float(d.get("avg_filtered_yolo_count", 0.0)),
        avg_raw_sahi_count=float(d.get("avg_raw_sahi_count", 0.0)),
        avg_accepted_sahi_count=float(d.get("avg_accepted_sahi_count", 0.0)),
        avg_raw_sam3_text_count=float(d.get("avg_raw_sam3_text_count", 0.0)),
        avg_accepted_sam3_text_count=float(d.get("avg_accepted_sam3_text_count", 0.0)),
        avg_mask_count=float(d.get("avg_mask_count", 0.0)),
        avg_final_prediction_count=float(d.get("avg_final_prediction_count", 0.0)),
    )


CHECKPOINT_FILENAME = "benchmark_checkpoint.jsonl"


def benchmark_checkpoint_path(mode_dir: Path) -> Path:
    return mode_dir / CHECKPOINT_FILENAME


def load_benchmark_checkpoint(
    mode_dir: Path,
) -> tuple[List[ImageMetrics], List[float], List[float], str, set[str]]:
    path = benchmark_checkpoint_path(mode_dir)
    per_image: List[ImageMetrics] = []
    map50_vals: List[float] = []
    map50_95_vals: List[float] = []
    notes = ""
    done: set[str] = set()
    if not path.is_file():
        return per_image, map50_vals, map50_95_vals, notes, done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        im = ImageMetrics(**rec["metrics"])
        per_image.append(im)
        done.add(im.image)
        if rec.get("map50") is not None:
            map50_vals.append(float(rec["map50"]))
        if rec.get("map50_95") is not None:
            map50_95_vals.append(float(rec["map50_95"]))
        if rec.get("notes"):
            notes = str(rec["notes"])
    return per_image, map50_vals, map50_95_vals, notes, done


def append_benchmark_checkpoint(
    mode_dir: Path,
    im: ImageMetrics,
    map50: float | None,
    map50_95: float | None,
    notes: str,
) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)
    path = benchmark_checkpoint_path(mode_dir)
    rec = {
        "image": im.image,
        "metrics": asdict(im),
        "map50": map50,
        "map50_95": map50_95,
        "notes": notes,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def clear_benchmark_checkpoint(mode_dir: Path) -> None:
    path = benchmark_checkpoint_path(mode_dir)
    if path.is_file():
        path.unlink()


def mode_benchmark_complete(mode_dir: Path, expected_images: int) -> bool:
    agg = load_mode_aggregate_from_metrics(mode_dir)
    return agg is not None and agg.images_evaluated >= expected_images


def save_mode_outputs(
    out_dir: Path,
    aggregate: ModeAggregate,
    map50: float | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate
    if map50 is not None:
        agg.map50 = map50
    metrics = {
        "mode": agg.mode,
        "images_evaluated": agg.images_evaluated,
        "precision": agg.precision,
        "recall": agg.recall,
        "f1": agg.f1,
        "accuracy": agg.accuracy,
        "map50": agg.map50,
        "map50_95": agg.map50_95,
        "iou_threshold": agg.iou_threshold,
        "true_positives": agg.total_tp,
        "false_positives": agg.total_fp,
        "false_negatives": agg.total_fn,
        "total_predictions": agg.total_predictions,
        "total_ground_truth": agg.total_ground_truth,
        "small_object_recall": agg.small_object_recall,
        "fps": agg.fps,
        "avg_inference_time": agg.avg_inference_time,
        "total_runtime_seconds": agg.total_runtime,
        "duplicate_detection_count": agg.duplicate_detection_count,
        "mask_quality": agg.mask_quality,
        "notes": agg.notes,
        "eval_conf": EVAL_CONF,
        "eval_iou": EVAL_IOU,
        "small_object_area_frac": SMALL_OBJECT_AREA_FRAC,
        "avg_raw_yolo_count": agg.avg_raw_yolo_count,
        "avg_filtered_yolo_count": agg.avg_filtered_yolo_count,
        "avg_raw_sahi_count": agg.avg_raw_sahi_count,
        "avg_accepted_sahi_count": agg.avg_accepted_sahi_count,
        "avg_raw_sam3_text_count": agg.avg_raw_sam3_text_count,
        "avg_accepted_sam3_text_count": agg.avg_accepted_sam3_text_count,
        "avg_mask_count": agg.avg_mask_count,
        "avg_final_prediction_count": agg.avg_final_prediction_count,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    import csv
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "precision", "recall", "f1", "accuracy", "tp", "fp", "fn",
            "gt_count", "pred_count", "small_object_recall", "mask_quality",
            "duplicate_pairs", "runtime_s",
        ])
        for r in agg.per_image:
            w.writerow([
                r.image, f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.f1:.4f}",
                f"{r.accuracy:.4f}", r.tp, r.fp, r.fn, r.ground_truth_count,
                r.predicted_count, f"{r.small_object_recall:.4f}", f"{r.mask_quality:.4f}",
                r.duplicate_pairs, f"{r.runtime_seconds:.4f}",
            ])


def write_supervisor_report_csv(
    path: Path,
    dataset: str,
    aggregates: Dict[str, ModeAggregate],
) -> None:
    """Full supervisor metrics table (one row per pipeline)."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset", "pipeline", "images_evaluated",
        "precision", "recall", "f1", "accuracy",
        "map50", "map50_95", "iou_threshold",
        "true_positives", "false_positives", "false_negatives",
        "total_predictions", "total_ground_truth_objects",
        "avg_fps", "avg_runtime_per_image_s", "total_runtime_s",
        "small_object_recall", "mask_quality", "duplicate_detection_count",
        "eval_confidence_threshold", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for mode in ALL_PIPELINE_ORDER:
            if mode not in aggregates:
                continue
            a = aggregates[mode]
            w.writerow([
                dataset, mode, a.images_evaluated,
                f"{a.precision:.6f}", f"{a.recall:.6f}", f"{a.f1:.6f}", f"{a.accuracy:.6f}",
                f"{a.map50:.6f}", f"{a.map50_95:.6f}", f"{a.iou_threshold:.2f}",
                a.total_tp, a.total_fp, a.total_fn,
                a.total_predictions, a.total_ground_truth,
                f"{a.fps:.4f}", f"{a.avg_inference_time:.4f}", f"{a.total_runtime:.2f}",
                f"{a.small_object_recall:.6f}", f"{a.mask_quality:.6f}", a.duplicate_detection_count,
                EVAL_CONF, a.notes,
            ])


ALL_PIPELINE_ORDER = (
    "yolo_only", "yolo_sahi", "sam3_only", "sam3_yolo", "yolo_sahi_sam3",
)


def write_comparison_csv(path: Path, aggregates: Dict[str, ModeAggregate]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "mode", "mAP50", "mAP50-95", "precision", "recall", "f1", "accuracy",
            "iou_threshold", "TP", "FP", "FN", "total_predictions", "total_gt",
            "small_object_recall", "mask_quality", "fps", "avg_inference_time",
            "total_runtime_s", "duplicate_detection_count", "notes",
        ])
        for mode in ALL_PIPELINE_ORDER:
            if mode not in aggregates:
                continue
            a = aggregates[mode]
            w.writerow([
                mode, f"{a.map50:.4f}", f"{a.map50_95:.4f}",
                f"{a.precision:.4f}", f"{a.recall:.4f}", f"{a.f1:.4f}", f"{a.accuracy:.4f}",
                f"{a.iou_threshold:.2f}", a.total_tp, a.total_fp, a.total_fn,
                a.total_predictions, a.total_ground_truth,
                f"{a.small_object_recall:.4f}", f"{a.mask_quality:.4f}",
                f"{a.fps:.2f}", f"{a.avg_inference_time:.4f}", f"{a.total_runtime:.2f}",
                a.duplicate_detection_count, a.notes,
            ])
