"""
Shared detection pipeline helpers — YOLO / SAHI / SAM3 text / SAM3 box masks.

SAM3 has two roles:
  1. sam3_text  — adds detections (text prompts)
  2. box masks  — segmentation overlay only; never changes boxes or predicted_count
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.pipeline_config import (
    CONTAINMENT_THRESHOLD,
    FINAL_NMS_IOU,
    MAX_ASPECT_RATIO,
    MAX_BOX_AREA_RATIO,
    MAX_BOX_MASKS,
    MAX_SAM3_TEXT_DETECTIONS_PER_CLASS,
    MAX_TOTAL_SAM3_TEXT_DETECTIONS,
    MERGE_IOU_THRESHOLD,
    MIN_ASPECT_RATIO,
    MIN_BOX_AREA,
    SAHI_CONF,
    SAHI_IOU,
    SAHI_LOW_CONF_FOR_MASK_DROP,
    SAHI_MASK_MIN_FILL,
    SAHI_OVERLAP,
    SAHI_POSTPROCESS,
    SAHI_SLICE,
    SAHI_YOLO_IMGSZ,
    SAM3_TEXT_CONF,
    SAM3_TEXT_NOT_IMPLEMENTED_MSG,
    SOURCE_PRIORITY,
    YOLO_CONF,
    YOLO_IMGSZ,
)
from core.sam3_support import resolve_sam3_weights, segment_with_text, segment_with_yolo_boxes
from pipelines._common import PipelineStats


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box1: List[float], box2: List[float]) -> float:
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(box1) + box_area(box2) - inter
    return inter / union if union > 0 else 0.0


def containment_ratio(inner_box: List[float], outer_box: List[float]) -> float:
    ia = box_area(inner_box)
    if ia <= 0:
        return 0.0
    ix1 = max(inner_box[0], outer_box[0])
    iy1 = max(inner_box[1], outer_box[1])
    ix2 = min(inner_box[2], outer_box[2])
    iy2 = min(inner_box[3], outer_box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / ia


def is_nested(
    candidate: List[float],
    existing_boxes: List[List[float]],
    *,
    contain_thresh: float = CONTAINMENT_THRESHOLD,
) -> bool:
    return any(containment_ratio(candidate, ob) >= contain_thresh for ob in existing_boxes)


def _aspect_ratio(box: List[float]) -> float:
    w = max(1.0, box[2] - box[0])
    h = max(1.0, box[3] - box[1])
    return min(w / h, h / w)


def passes_box_filters(
    det: Dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    min_conf: float,
) -> bool:
    score = float(det.get("score", 0.0))
    if score < min_conf:
        return False
    bbox = det["bbox"]
    area = box_area(bbox)
    if area < MIN_BOX_AREA:
        return False
    img_area = float(image_width * image_height)
    if area > MAX_BOX_AREA_RATIO * img_area:
        return False
    ar = _aspect_ratio(bbox)
    if ar < MIN_ASPECT_RATIO or ar > MAX_ASPECT_RATIO:
        return False
    return True


def normalize_detection(
    det: Dict[str, Any],
    source: str,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    cid = int(det.get("class_id", 0))
    name = det.get("class_name") or (class_names or {}).get(cid, str(cid))
    src = source
    if source == "yolo_sahi":
        src = "sahi"
    return {
        "bbox": [float(v) for v in det["bbox"]],
        "class_id": cid,
        "class_name": str(name),
        "score": float(det.get("score", 0.0)),
        "source": src,
    }


def text_prompts_from_class_names(class_names: Optional[Dict[int, str]]) -> List[str]:
    if not class_names:
        return []
    return [str(class_names[i]) for i in sorted(class_names.keys())]


# ---------------------------------------------------------------------------
# NMS / merge / nested removal
# ---------------------------------------------------------------------------

def _source_rank(det: Dict[str, Any]) -> int:
    return SOURCE_PRIORITY.get(str(det.get("source", "")), 0)


def class_aware_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = FINAL_NMS_IOU,
    *,
    masks: Optional[List[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Any], int]:
    """Greedy NMS; higher source priority wins, then higher score."""
    if len(detections) <= 1:
        m = masks if masks is not None else [None] * len(detections)
        return detections, m, 0

    order = sorted(
        range(len(detections)),
        key=lambda i: (_source_rank(detections[i]), float(detections[i].get("score", 0.0))),
        reverse=True,
    )
    kept: List[int] = []
    removed = 0
    for idx in order:
        dup = False
        for k in kept:
            if detections[idx].get("class_id") != detections[k].get("class_id"):
                continue
            if iou(detections[idx]["bbox"], detections[k]["bbox"]) >= iou_threshold:
                dup = True
                break
        if dup:
            removed += 1
        else:
            kept.append(idx)
    kept.sort()
    out_d = [detections[i] for i in kept]
    out_m: List[Any] = (
        [masks[i] for i in kept]
        if masks is not None and len(masks) == len(detections)
        else [None] * len(out_d)
    )
    return out_d, out_m, removed


def remove_nested_boxes(
    detections: List[Dict[str, Any]],
    *,
    contain_thresh: float = CONTAINMENT_THRESHOLD,
    masks: Optional[List[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Any], int]:
    """Drop lower-priority/score boxes nested inside a kept box (same class)."""
    if not detections:
        return [], [], 0
    if masks is None:
        masks = [None] * len(detections)

    order = sorted(
        range(len(detections)),
        key=lambda i: (_source_rank(detections[i]), float(detections[i].get("score", 0.0))),
        reverse=True,
    )
    kept: List[int] = []
    removed = 0
    for idx in order:
        nested = False
        for k in kept:
            if detections[idx].get("class_id") != detections[k].get("class_id"):
                continue
            if containment_ratio(detections[idx]["bbox"], detections[k]["bbox"]) >= contain_thresh:
                nested = True
                break
        if nested:
            removed += 1
        else:
            kept.append(idx)
    kept.sort()
    return [detections[i] for i in kept], [masks[i] for i in kept], removed


def _conflicts_with_pool(
    det: Dict[str, Any],
    pool: List[Dict[str, Any]],
) -> bool:
    cid = int(det.get("class_id", -1))
    bbox = det["bbox"]
    for ex in pool:
        if int(ex.get("class_id", -1)) != cid:
            continue
        if iou(bbox, ex["bbox"]) >= MERGE_IOU_THRESHOLD:
            return True
        if is_nested(bbox, [ex["bbox"]]):
            return True
    return False


def merge_with_priority(
    base_detections: List[Dict[str, Any]],
    candidate_detections: List[Dict[str, Any]],
    source_name: str,
    *,
    image_width: int,
    image_height: int,
    min_conf: float,
    class_names: Optional[Dict[int, str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Add candidates only when they are truly missing (no same-class IoU/nest conflict).
    Returns (merged_list, accepted_count).
    """
    merged = list(base_detections)
    pool = list(base_detections)
    accepted = 0
    for raw in candidate_detections:
        det = normalize_detection(raw, source_name, class_names)
        if not passes_box_filters(det, image_width, image_height, min_conf=min_conf):
            continue
        if _conflicts_with_pool(det, pool):
            continue
        merged.append(det)
        pool.append(det)
        accepted += 1
    return merged, accepted


def cap_sam3_text_detections(
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per-class and global caps on SAM3 text additions."""
    by_class: Dict[int, List[Dict[str, Any]]] = {}
    for det in detections:
        if det.get("source") != "sam3_text":
            continue
        cid = int(det["class_id"])
        by_class.setdefault(cid, []).append(det)

    capped_sam3: List[Dict[str, Any]] = []
    for cid in sorted(by_class.keys()):
        cls_dets = sorted(by_class[cid], key=lambda d: d.get("score", 0.0), reverse=True)
        capped_sam3.extend(cls_dets[:MAX_SAM3_TEXT_DETECTIONS_PER_CLASS])

    capped_sam3 = sorted(capped_sam3, key=lambda d: d.get("score", 0.0), reverse=True)
    capped_sam3 = capped_sam3[:MAX_TOTAL_SAM3_TEXT_DETECTIONS]

    non_sam3 = [d for d in detections if d.get("source") != "sam3_text"]
    return non_sam3 + capped_sam3


def filter_detections_stage(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    min_conf: float,
) -> List[Dict[str, Any]]:
    return [
        d for d in detections
        if passes_box_filters(d, image_width, image_height, min_conf=min_conf)
    ]


# ---------------------------------------------------------------------------
# Detector runners
# ---------------------------------------------------------------------------

def run_yolo_detection(
    image_bgr: np.ndarray,
    yolo_model: Any,
    device: str,
    class_names: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    results = yolo_model.predict(
        source=image_bgr,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        device=device,
        verbose=False,
    )
    dets: List[Dict[str, Any]] = []
    if not results or len(results) == 0:
        return dets
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return dets
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    scores = r.boxes.conf.cpu().numpy()
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        det = normalize_detection(
            {"bbox": [x1, y1, x2, y2], "class_id": int(cls[i]), "score": float(scores[i])},
            "yolo",
            class_names,
        )
        dets.append(det)
    return dets


def run_sahi_yolo_detection(
    image_bgr: np.ndarray,
    yolo_weights: str,
    device: str,
    class_names: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    from sahi.predict import get_sliced_prediction

    from core.sahi_model_cache import get_sahi_detection_model

    sahi_model = get_sahi_detection_model(
        yolo_weights,
        device,
        confidence_threshold=SAHI_CONF,
        image_size=SAHI_YOLO_IMGSZ,
    )
    result = get_sliced_prediction(
        image_bgr,
        sahi_model,
        slice_height=SAHI_SLICE,
        slice_width=SAHI_SLICE,
        overlap_height_ratio=SAHI_OVERLAP,
        overlap_width_ratio=SAHI_OVERLAP,
        postprocess_type=SAHI_POSTPROCESS,
        postprocess_match_metric="IOU",
        postprocess_match_threshold=SAHI_IOU,
        verbose=0,
    )
    dets: List[Dict[str, Any]] = []
    for pred in result.object_prediction_list:
        score = float(pred.score.value)
        x1 = float(pred.bbox.minx)
        y1 = float(pred.bbox.miny)
        x2 = float(pred.bbox.maxx)
        y2 = float(pred.bbox.maxy)
        det = normalize_detection(
            {
                "bbox": [x1, y1, x2, y2],
                "class_id": int(pred.category.id),
                "score": score,
            },
            "sahi",
            class_names,
        )
        dets.append(det)
    return dets


def run_sam3_text_detection(
    image_bgr: np.ndarray,
    class_names: Optional[Dict[int, str]],
    device: str,
    *,
    sam3_weights: Any = None,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """SAM3 text detection — once per image, batch prompts. Raises if API unavailable."""
    prompts = text_prompts_from_class_names(class_names)
    if not prompts:
        return [], []

    weights = sam3_weights or resolve_sam3_weights()
    try:
        raw_dets, raw_masks = segment_with_text(
            image_bgr,
            prompts,
            weights=weights,
            conf=SAM3_TEXT_CONF,
            device=device,
        )
    except NotImplementedError:
        raise
    except Exception as exc:
        raise NotImplementedError(SAM3_TEXT_NOT_IMPLEMENTED_MSG) from exc

    dets: List[Dict[str, Any]] = []
    for raw in raw_dets:
        dets.append(normalize_detection(raw, "sam3_text", class_names))
    return dets, raw_masks


def run_sam3_box_masks(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    device: str,
    *,
    sam3_weights: Any = None,
) -> Tuple[List[Dict[str, Any]], List[Any], int]:
    """
    SAM3 box masking only — never adds detections.
    Returns (detections_unchanged_or_sahi_filtered, aligned_masks, mask_count).
    """
    if not detections:
        return [], [], 0

    h, w = image_bgr.shape[:2]
    order = sorted(
        range(len(detections)),
        key=lambda i: float(detections[i].get("score", 0.0)),
        reverse=True,
    )
    mask_slots = set(order[:MAX_BOX_MASKS])

    to_mask = [detections[i] for i in sorted(mask_slots)]
    weights = sam3_weights or resolve_sam3_weights()

    try:
        _, mask_tensors = segment_with_yolo_boxes(
            image_bgr,
            to_mask,
            weights=weights,
            conf=SAM3_TEXT_CONF,
            device=device,
            drop_unsegmentable=False,
            min_mask_in_box_frac=SAHI_MASK_MIN_FILL,
        )
    except Exception:
        mask_tensors = [None] * len(to_mask)

    mask_by_idx: Dict[int, Any] = {}
    for slot, mi in enumerate(sorted(mask_slots)):
        mask_by_idx[mi] = mask_tensors[slot] if slot < len(mask_tensors) else None

    final_dets: List[Dict[str, Any]] = []
    final_masks: List[Any] = []
    mask_count = 0

    for i, det in enumerate(detections):
        src = str(det.get("source", ""))
        if i not in mask_slots:
            final_dets.append(det)
            final_masks.append(None)
            continue

        mask = mask_by_idx.get(i)
        valid_mask = False
        if mask is not None and getattr(mask, "size", 0):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            ba = box_area(det["bbox"])
            fill = float(mask[y1:y2, x1:x2].sum()) / max(ba, 1.0)
            valid_mask = fill >= SAHI_MASK_MIN_FILL

        if src == "sahi" and not valid_mask and float(det.get("score", 0)) < SAHI_LOW_CONF_FOR_MASK_DROP:
            continue  # drop suspicious SAHI-only without mask

        final_dets.append(det)
        final_masks.append(mask if valid_mask else None)
        if valid_mask:
            mask_count += 1

    return final_dets, final_masks, mask_count


def finalize_detections(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    masks: Optional[List[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Any], int]:
    """Final nested removal + class-aware NMS."""
    dets, ms, _ = remove_nested_boxes(detections, masks=masks)
    dets, ms, _ = class_aware_nms(dets, FINAL_NMS_IOU, masks=ms)
    return dets, ms, len(dets)


def log_pipeline_stats(stats: PipelineStats, *, prefix: str = "") -> None:
    p = f"{prefix} " if prefix else ""
    print(
        f"{p}[pipeline] yolo raw={stats.raw_yolo_count} filt={stats.filtered_yolo_count} | "
        f"sahi raw={stats.raw_sahi_count} acc={stats.accepted_sahi_count} | "
        f"sam3_text raw={stats.raw_sam3_text_count} acc={stats.accepted_sam3_text_count} | "
        f"final={stats.final_prediction_count} masks={stats.mask_count}"
    )
