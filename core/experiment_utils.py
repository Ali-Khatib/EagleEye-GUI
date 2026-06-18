from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


NA = "N/A"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def yolo_label_to_xyxy(
    cls: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, float, float, float, float, bool]:
    """Convert normalized YOLO xywh labels to clamped pixel xyxy boxes."""
    raw_x1 = (x_center - width / 2.0) * image_width
    raw_y1 = (y_center - height / 2.0) * image_height
    raw_x2 = (x_center + width / 2.0) * image_width
    raw_y2 = (y_center + height / 2.0) * image_height

    x1 = clamp(raw_x1, 0.0, float(image_width - 1))
    y1 = clamp(raw_y1, 0.0, float(image_height - 1))
    x2 = clamp(raw_x2, 0.0, float(image_width - 1))
    y2 = clamp(raw_y2, 0.0, float(image_height - 1))

    was_clamped = (
        x1 != raw_x1 or y1 != raw_y1 or x2 != raw_x2 or y2 != raw_y2
    )
    return cls, x1, y1, x2, y2, was_clamped


def _expected_label_path(image_path: str) -> Path:
    p = Path(image_path)
    return p.with_suffix(".txt")


def _label_pair_warnings(image_path: str, label_path: str) -> List[str]:
    image = Path(image_path)
    label = Path(label_path)
    warnings: List[str] = []

    expected = _expected_label_path(image_path)
    if image.as_posix().replace("\\", "/") == "data/demo/test_image.jpg":
        strict = Path("data/demo/test_image.txt")
        if label != strict:
            warnings.append(
                "IMAGE_PATH is data/demo/test_image.jpg, so LABEL_PATH must be "
                "data/demo/test_image.txt."
            )
    elif image.parent == label.parent and image.stem != label.stem:
        warnings.append(
            f"Image/label stem mismatch: image '{image.name}' vs label "
            f"'{label.name}'. Expected '{expected.name}'."
        )
    return warnings


def load_ground_truth_yolo_labels_checked(
    label_path: str,
    image_width: int,
    image_height: int,
    image_path: str | None = None,
) -> Dict[str, Any]:
    """Load YOLO GT labels, clamp to image, and return reliability warnings."""
    label = Path(label_path)
    warnings: List[str] = []
    boxes: List[Dict[str, Any]] = []
    class_ids: List[int] = []
    invalid_lines = 0
    clamped_boxes = 0
    impossible_boxes = 0
    tiny_boxes = 0

    if image_path:
        warnings.extend(_label_pair_warnings(image_path, label_path))

    if not label.is_file():
        warnings.append(f"Ground-truth label file is missing: {label_path}")
        return {
            "boxes": [],
            "warnings": warnings,
            "reliable": False,
            "class_ids": [],
        }

    with open(label, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                invalid_lines += 1
                warnings.append(f"Line {line_no}: expected 5 YOLO fields.")
                continue
            try:
                cls = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                invalid_lines += 1
                warnings.append(f"Line {line_no}: non-numeric YOLO label.")
                continue

            if width <= 0 or height <= 0:
                impossible_boxes += 1
                warnings.append(f"Line {line_no}: non-positive box size.")
                continue
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                warnings.append(f"Line {line_no}: center is outside [0, 1].")
            if not (0 < width <= 1 and 0 < height <= 1):
                warnings.append(f"Line {line_no}: width/height outside (0, 1].")

            cls, x1, y1, x2, y2, was_clamped = yolo_label_to_xyxy(
                cls, x_center, y_center, width, height, image_width, image_height
            )
            if was_clamped:
                clamped_boxes += 1
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if x2 <= x1 or y2 <= y1 or area <= 1:
                impossible_boxes += 1
                warnings.append(
                    f"Line {line_no}: impossible pixel box after conversion."
                )
                continue
            if area < 9:
                tiny_boxes += 1

            boxes.append({"class_id": cls, "bbox": [x1, y1, x2, y2]})
            class_ids.append(cls)

    if not boxes:
        warnings.append("Ground-truth label file has zero usable labels.")

    total_seen = len(boxes) + impossible_boxes
    if clamped_boxes:
        warnings.append(f"{clamped_boxes} GT boxes were clamped to image bounds.")
    if invalid_lines:
        warnings.append(f"{invalid_lines} invalid label lines were skipped.")
    if impossible_boxes:
        warnings.append(f"{impossible_boxes} impossible boxes were skipped.")
    if boxes and tiny_boxes / len(boxes) > 0.8:
        warnings.append(
            "Most GT boxes are tiny (<9 px^2); labels may use the wrong image size."
        )

    reliable = bool(boxes)
    if total_seen and impossible_boxes / total_seen > 0.2:
        reliable = False
    if boxes and tiny_boxes / len(boxes) > 0.8:
        reliable = False
    if any("mismatch" in w or "must be" in w for w in warnings):
        reliable = False

    return {
        "boxes": boxes,
        "warnings": warnings,
        "reliable": reliable,
        "class_ids": sorted(set(class_ids)),
    }


def load_prediction_yolo_labels(
    label_path: str,
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    """Load YOLO-format prediction txt (optional 6th field = confidence score)."""
    label = Path(label_path)
    if not label.is_file():
        return []

    detections: List[Dict[str, Any]] = []
    with open(label, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                score = float(parts[5]) if len(parts) >= 6 else 1.0
            except ValueError:
                continue
            if width <= 0 or height <= 0:
                continue
            cls, x1, y1, x2, y2, _ = yolo_label_to_xyxy(
                cls, x_center, y_center, width, height, image_width, image_height,
            )
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cls,
                "score": score,
            })
    return detections


def mask_to_xyxy(mask: Any) -> List[float] | None:
    """Return the tight (x1, y1, x2, y2) of a non-zero binary mask, or None.

    Used by YOLO+SAM and SAHI+YOLO+SAM to replace each YOLO/SAHI box with the
    tight bounding box of its SAM 2.1 mask. SAM masks are usually much tighter
    than detector boxes, which improves IoU vs ground truth at threshold 0.5.
    """
    if mask is None:
        return None
    try:
        import numpy as np  # local import to keep this module dependency-light
    except Exception:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 0 or arr.size == 0:
        return None
    if arr.dtype == bool:
        nz = arr
    else:
        nz = arr > 0
    ys, xs = np.where(nz)
    if xs.size == 0 or ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max()), float(ys.max())]


def refine_detections_with_masks(
    detections: List[Dict[str, Any]],
    masks: List[Any],
    *,
    min_mask_pixels: int = 8,
    min_area_ratio: float = 0.02,
    drop_empty: bool = False,
    drop_tiny: bool = False,
) -> Dict[str, Any]:
    """
    Replace each detection's bbox with the tight bbox of its SAM mask when SAM
    succeeds. By default, keep the original detector box if SAM fails (avoids
    killing recall). Set drop_empty / drop_tiny True to filter false positives.

    Returns:
        {
            "detections": [...refined dets, with bbox replaced and
                            sam_refined=True...],
            "masks":      [...masks aligned with returned detections...],
            "dropped_empty":  int,  # SAM produced zero mask (and dropped)
            "dropped_tiny":   int,  # mask far smaller than YOLO box (and dropped)
            "kept":           int,  # bbox replaced from SAM mask
            "kept_original":  int,  # SAM failed but detector box kept
        }
    """
    refined: List[Dict[str, Any]] = []
    kept_masks: List[Any] = []
    dropped_empty = 0
    dropped_tiny = 0
    kept_original = 0
    for det, mask in zip(detections, masks):
        bbox = mask_to_xyxy(mask)
        if bbox is None:
            if drop_empty:
                dropped_empty += 1
                continue
            refined.append({**det, "sam_refined": False})
            kept_masks.append(mask)
            kept_original += 1
            continue
        x1, y1, x2, y2 = bbox
        sam_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if sam_area < min_mask_pixels:
            if drop_empty:
                dropped_empty += 1
                continue
            refined.append({**det, "sam_refined": False})
            kept_masks.append(mask)
            kept_original += 1
            continue
        det_box = det.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        yx1, yy1, yx2, yy2 = det_box
        yolo_area = max(1.0, (yx2 - yx1) * (yy2 - yy1))
        if (sam_area / yolo_area) < min_area_ratio:
            if drop_tiny:
                dropped_tiny += 1
                continue
            refined.append({**det, "sam_refined": False})
            kept_masks.append(mask)
            kept_original += 1
            continue
        refined.append({
            **det,
            "bbox": [x1, y1, x2, y2],
            "sam_refined": True,
        })
        kept_masks.append(mask)
    kept = sum(1 for d in refined if d.get("sam_refined"))
    return {
        "detections": refined,
        "masks": kept_masks,
        "dropped_empty": dropped_empty,
        "dropped_tiny": dropped_tiny,
        "kept": kept,
        "kept_original": kept_original,
    }


def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def merge_amg_proposals_after_refined(
    refined_detections: List[Dict[str, Any]],
    refined_masks: List[Any],
    amg_masks_data: List[Dict[str, Any]],
    *,
    merge_iou_thresh: float = 0.5,
    min_area: float = 250.0,
) -> Tuple[List[Dict[str, Any]], List[Any], int]:
    """Append AMG masks whose tight box is not already covered by refined boxes.

    Used after SAHI+YOLO boxes are SAM-refined: SAM Automatic Mask Generator
    runs on the full image and adds ``class_id=-1`` proposals (``source='sam_amg'``)
    that do not overlap any existing box by ``merge_iou_thresh`` IoU.
    """
    import numpy as np

    merged_dets: List[Dict[str, Any]] = [dict(d) for d in refined_detections]
    merged_masks: List[Any] = list(refined_masks)
    accepted_boxes: List[List[float]] = [
        [float(x) for x in det["bbox"]] for det in merged_dets
    ]

    added = 0
    sorted_amg = sorted(
        amg_masks_data,
        key=lambda m: float(m.get("predicted_iou", 0.0)),
        reverse=True,
    )
    for m in sorted_amg:
        seg = m.get("segmentation")
        if seg is None:
            continue
        arr = np.asarray(seg)
        if arr.size == 0:
            continue
        binm = (arr > 0).astype(np.uint8) if arr.dtype != bool else arr.astype(np.uint8)
        bbox = mask_to_xyxy(binm)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < min_area:
            continue
        best_iou = max(
            (calculate_iou(bbox, ob) for ob in accepted_boxes),
            default=0.0,
        )
        if best_iou >= merge_iou_thresh:
            continue
        score = float(m.get("predicted_iou", 0.0))
        merged_dets.append(
            {
                "bbox": bbox,
                "class_id": -1,
                "score": score,
                "source": "sam_amg",
            }
        )
        merged_masks.append(binm)
        accepted_boxes.append([float(x1), float(y1), float(x2), float(y2)])
        added += 1

    return merged_dets, merged_masks, added


def evaluate_detections(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_thresh: float,
    class_agnostic: bool = False,
) -> Dict[str, Any]:
    """One-to-one IoU matching. Class-aware unless class_agnostic=True."""
    tp = 0
    fp = 0
    used_gt = [False] * len(ground_truth)
    preds_sorted = sorted(
        predictions, key=lambda d: d.get("score", 0.0), reverse=True
    )

    for pred in preds_sorted:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(ground_truth):
            if used_gt[idx]:
                continue
            if not class_agnostic and pred.get("class_id") != gt.get("class_id"):
                continue
            iou = calculate_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thresh:
            tp += 1
            used_gt[best_idx] = True
        else:
            fp += 1

    fn = used_gt.count(False)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_with_ground_truth(
    predictions: List[Dict[str, Any]],
    label_path: str,
    image_width: int,
    image_height: int,
    image_path: str,
    iou_thresh: float,
    class_agnostic: bool,
    base_notes: str = "",
) -> Dict[str, Any]:
    """Load GT, validate it, and return metrics or N/A if unreliable."""
    gt_result = load_ground_truth_yolo_labels_checked(
        label_path, image_width, image_height, image_path=image_path
    )
    gt = gt_result["boxes"]
    warnings = gt_result["warnings"]
    notes: List[str] = []
    if base_notes:
        notes.append(base_notes)
    notes.extend(warnings)

    if not gt_result["reliable"]:
        return {
            "precision": NA,
            "recall": NA,
            "f1": NA,
            "accuracy": NA,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ground_truth_count": len(gt),
            "notes": " ".join(notes) if notes else "Ground truth is not reliable.",
            "warnings": warnings,
            "has_reliable_gt": False,
        }

    metrics = evaluate_detections(predictions, gt, iou_thresh, class_agnostic)
    return {
        **metrics,
        "ground_truth_count": len(gt),
        "notes": " ".join(notes),
        "warnings": warnings,
        "has_reliable_gt": True,
    }


def metric_or_na(value: Any, digits: int = 4) -> Any:
    if value == NA:
        return NA
    return round(float(value), digits)
