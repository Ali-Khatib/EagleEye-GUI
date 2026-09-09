"""Shared detection helpers: SAHI parse, NMS, YOLO conversion."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.eval_config import (
    EVAL_CONF,
    MIN_BOX_AREA_PX,
    SMALL_OBJECT_AREA_FRAC,
    STACK_CONTAIN_FRAC,
    STACK_DEDUP_IOU,
    STACK_MERGE_IOU,
    STACK_MIN_SAM_MASK_FILL,
    STACK_MIN_SCORE,
    STACK_MIN_SCORE_PEDESTRIAN,
    STACK_SAM3_TEXT_CONF,
)
from core.experiment_utils import calculate_iou


def box_area(bbox: List[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def fraction_inside(inner: List[float], outer: List[float]) -> float:
    """Share of inner box area that lies inside outer (1.0 = fully contained)."""
    ia = box_area(inner)
    if ia <= 0:
        return 0.0
    return intersection_area(inner, outer) / ia


def _duplicate_of_kept(
    candidate: Dict[str, Any],
    kept_det: Dict[str, Any],
    *,
    iou_thresh: float,
    contain_frac: float,
    class_aware: bool,
) -> bool:
    if class_aware and candidate.get("class_id") != kept_det.get("class_id"):
        return False
    cb = candidate["bbox"]
    kb = kept_det["bbox"]
    if calculate_iou(cb, kb) >= iou_thresh:
        return True
    if contain_frac > 0 and fraction_inside(cb, kb) >= contain_frac:
        return True
    return False


def deduplicate_boxes(
    detections: List[Dict[str, Any]],
    iou_thresh: float,
    *,
    class_aware: bool = True,
    masks: Optional[List[Any]] = None,
    contain_frac: float = 0.0,
) -> Tuple[List[Dict[str, Any]], int, Optional[List[Any]]]:
    """Greedy NMS by score; optional containment drop for nested mini-boxes."""
    if len(detections) <= 1:
        return detections, 0, masks
    order = sorted(
        range(len(detections)),
        key=lambda i: float(detections[i].get("score", 0.0)),
        reverse=True,
    )
    kept: List[int] = []
    removed = 0
    for idx in order:
        dup = False
        for k in kept:
            if _duplicate_of_kept(
                detections[idx],
                detections[k],
                iou_thresh=iou_thresh,
                contain_frac=contain_frac,
                class_aware=class_aware,
            ):
                dup = True
                break
        if dup:
            removed += 1
        else:
            kept.append(idx)
    kept.sort()
    out_masks = [masks[i] for i in kept] if masks is not None and len(masks) == len(detections) else masks
    return [detections[i] for i in kept], removed, out_masks


def is_small_box(bbox: List[float], img_w: int, img_h: int, frac: float) -> bool:
    return box_area(bbox) < frac * float(img_w * img_h)


def _overlaps_pool(
    bbox: List[float],
    class_id: int,
    pool: List[Dict[str, Any]],
    *,
    iou_thresh: float,
    contain_frac: float,
) -> bool:
    for det in pool:
        if int(det.get("class_id", -1)) != class_id:
            continue
        if _duplicate_of_kept(
            {"bbox": bbox, "class_id": class_id},
            det,
            iou_thresh=iou_thresh,
            contain_frac=contain_frac,
            class_aware=True,
        ):
            return True
    return False


def merge_yolo_sahi_detections(
    yolo_dets: List[Dict[str, Any]],
    sahi_dets: List[Dict[str, Any]],
    *,
    img_w: int,
    img_h: int,
    iou_thresh: float = STACK_MERGE_IOU,
    small_frac: float = SMALL_OBJECT_AREA_FRAC,
) -> List[Dict[str, Any]]:
    """
    YOLO backbone + small SAHI extras only where YOLO missed (no overlap).
    Large SAHI slice hits are usually duplicates or texture FPs — skip them.
    """
    merged: List[Dict[str, Any]] = [
        {**det, "source": det.get("source", "yolo")} for det in yolo_dets
    ]
    for det in sahi_dets:
        if not is_small_box(det["bbox"], img_w, img_h, small_frac):
            continue
        cid = int(det.get("class_id", -1))
        if _overlaps_pool(
            det["bbox"],
            cid,
            yolo_dets,
            iou_thresh=iou_thresh,
            contain_frac=STACK_CONTAIN_FRAC,
        ):
            continue
        merged.append({**det, "source": det.get("source", "yolo_sahi")})
    return merged


def _is_person_like_class(class_id: int, class_names: Optional[Dict[int, str]]) -> bool:
    if class_names:
        name = str(class_names.get(class_id, "")).lower()
        return any(k in name for k in ("pedestrian", "people", "person"))
    return class_id in {0, 1}


def text_prompts_from_class_names(
    class_names: Optional[Dict[int, str]],
) -> List[str]:
    """YOLO class names in id order — used as SAM3 text detection prompts."""
    if not class_names:
        return []
    return [str(class_names[i]) for i in sorted(class_names.keys())]


def merge_sam3_text_detections(
    base_dets: List[Dict[str, Any]],
    sam3_dets: List[Dict[str, Any]],
    sam3_masks: List[Any],
    *,
    iou_thresh: float = STACK_MERGE_IOU,
    min_score: float = STACK_SAM3_TEXT_CONF,
) -> Tuple[List[Dict[str, Any]], List[Any], int]:
    """
    SAM3 text-prompt detections that do not overlap existing boxes.
    Returns (extra_dets, extra_masks, added_count).
    """
    extras_d: List[Dict[str, Any]] = []
    extras_m: List[Any] = []
    pool = list(base_dets)
    added = 0
    for det, mask in zip(sam3_dets, sam3_masks):
        if float(det.get("score", 0.0)) < min_score:
            continue
        bbox = det["bbox"]
        cid = int(det.get("class_id", -1))
        if _overlaps_pool(
            bbox,
            cid,
            pool,
            iou_thresh=iou_thresh,
            contain_frac=STACK_CONTAIN_FRAC,
        ):
            continue
        entry = {
            **det,
            "source": "sam3_text",
            "sam3_refined": True,
        }
        extras_d.append(entry)
        extras_m.append(mask)
        pool.append(entry)
        added += 1
    return extras_d, extras_m, added


def filter_stack_quality(
    detections: List[Dict[str, Any]],
    masks: Optional[List[Any]],
    img_h: int,
    img_w: int,
    *,
    class_names: Optional[Dict[int, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Any], int]:
    """
    Tiered post-filter for YOLO+SAHI+SAM3:
    - Full-frame YOLO boxes always kept (reliable backbone).
    - SAM3 text-detected boxes kept when score passes (already have masks).
    - SAHI-only extras must pass score floor + SAM mask verification.
    """
    if not detections:
        return [], [], 0
    if masks is None:
        masks = [None] * len(detections)
    kept_d: List[Dict[str, Any]] = []
    kept_m: List[Any] = []
    dropped = 0
    for det, mask in zip(detections, masks):
        score = float(det.get("score", 0.0))
        cid = int(det.get("class_id", -1))
        source = str(det.get("source", "yolo_sahi"))

        if source == "yolo" and score >= EVAL_CONF:
            kept_d.append(det)
            kept_m.append(mask)
            continue

        if source == "sam3_text" and score >= STACK_SAM3_TEXT_CONF:
            kept_d.append(det)
            kept_m.append(mask)
            continue

        person_like = _is_person_like_class(cid, class_names)
        min_score = STACK_MIN_SCORE_PEDESTRIAN if person_like else STACK_MIN_SCORE
        if score < min_score:
            dropped += 1
            continue
        if not det.get("sam3_refined", False):
            dropped += 1
            continue
        if mask is not None and getattr(mask, "size", 0):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            x1, x2 = max(0, x1), min(img_w, x2)
            y1, y2 = max(0, y1), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                dropped += 1
                continue
            ba = box_area(det["bbox"])
            fill = float(mask[y1:y2, x1:x2].sum()) / max(ba, 1.0)
            if fill < STACK_MIN_SAM_MASK_FILL:
                dropped += 1
                continue
        kept_d.append(det)
        kept_m.append(mask)
    return kept_d, kept_m, dropped


def count_duplicate_predictions(
    detections: List[Dict[str, Any]],
    iou_thresh: float = 0.5,
) -> int:
    """Pairs of predictions with IoU >= thresh (lower-scored counted as duplicate)."""
    n = 0
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if calculate_iou(detections[i]["bbox"], detections[j]["bbox"]) >= iou_thresh:
                n += 1
    return n


def sahi_result_to_detections(
    result: Any,
    *,
    conf_floor: float = EVAL_CONF,
    min_area: float = MIN_BOX_AREA_PX,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for pred in result.object_prediction_list:
        score = float(pred.score.value)
        if score < conf_floor:
            continue
        x1 = float(pred.bbox.minx)
        y1 = float(pred.bbox.miny)
        x2 = float(pred.bbox.maxx)
        y2 = float(pred.bbox.maxy)
        if box_area([x1, y1, x2, y2]) < min_area:
            continue
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": int(pred.category.id),
            "score": score,
            "source": "yolo_sahi",
        })
    return detections


def yolo_results_to_detections(
    results: Any,
    *,
    conf_floor: float = EVAL_CONF,
    min_area: float = MIN_BOX_AREA_PX,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if not results or len(results) == 0:
        return detections
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return detections
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    scores = r.boxes.conf.cpu().numpy()
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        if float(scores[i]) < conf_floor:
            continue
        if box_area([x1, y1, x2, y2]) < min_area:
            continue
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": int(cls[i]),
            "score": float(scores[i]),
            "source": "yolo",
        })
    return detections


def detections_to_yolo_txt(
    detections: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> str:
    lines: List[str] = []
    for d in detections:
        cid = int(d.get("class_id", 0))
        if cid < 0:
            continue
        x1, y1, x2, y2 = d["bbox"]
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        score = float(d.get("score", 1.0))
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {score:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")
