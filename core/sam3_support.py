"""
SAM 3 via Ultralytics (Meta SAM 3 + SAM3SemanticPredictor / SAM box prompts).

Weights: download sam3.pt from https://huggingface.co/facebook/sam3 (license required).
Place at repo root as sam3.pt or set SAM3_WEIGHTS env var.

YOLO detects objects; SAHI improves small-object recall; SAM 3 refines masks using
text or box prompts — see pipelines/ and README.md.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.paths import PROJECT_ROOT

DEFAULT_SAM3_WEIGHTS = PROJECT_ROOT / "sam3.pt"

# Checked in order after SAM3_WEIGHTS env and explicit --sam3-weights
SAM3_SEARCH_PATHS = (
    DEFAULT_SAM3_WEIGHTS,
    PROJECT_ROOT / "models" / "sam3" / "sam3.pt",
    PROJECT_ROOT / "models" / "sam3.pt",
)


def infer_sam3_dataset(source: Path, yolo_weights: str, dataset_name: str = "") -> str | None:
    """Guess visdrone vs kitti for fine-tuned SAM3 checkpoint lookup."""
    blob = f"{source} {yolo_weights} {dataset_name}".lower()
    if "visdrone" in blob or "train9" in blob:
        return "visdrone"
    if "kitti" in blob or "train2" in blob:
        return "kitti"
    return None

_models: Dict[str, Any] = {}


def resolve_sam3_weights(explicit: str | None = None, dataset: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p.resolve()
        p2 = PROJECT_ROOT / explicit
        if p2.is_file():
            return p2.resolve()
    if dataset:
        ds = dataset.lower().strip()
        finetuned = (
            PROJECT_ROOT / "runs" / "sam3_finetune" / ds / "sam3.pt",
            PROJECT_ROOT / "runs" / "sam3_finetune" / ds / "checkpoint.pt",
        )
        for candidate in finetuned:
            if candidate.is_file():
                return candidate.resolve()
    env = os.environ.get("SAM3_WEIGHTS", "").strip()
    if env and Path(env).is_file():
        return Path(env).resolve()
    for candidate in SAM3_SEARCH_PATHS:
        if candidate.is_file():
            return candidate.resolve()
    # Hugging Face hub cache (if download_sam3.py was used)
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub:
        cached = Path(hub) / "models--facebook--sam3" / "snapshots"
        if cached.is_dir():
            for snap in sorted(cached.iterdir(), reverse=True):
                pt = snap / "sam3.pt"
                if pt.is_file():
                    return pt.resolve()
    raise FileNotFoundError(
        "SAM 3 weights not found. Download sam3.pt from "
        "https://huggingface.co/facebook/sam3 (request access first), then place at:\n"
        f"  {DEFAULT_SAM3_WEIGHTS}\n"
        "Or run: python scripts/download_sam3.py\n"
        "Or set SAM3_WEIGHTS to the full path of your .pt file."
    )


def _get_semantic_predictor(weights: Path, conf: float, device: str) -> Any:
    key = f"semantic_{weights}_{conf}_{device}"
    if key in _models:
        return _models[key]
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=str(weights),
        device=device,
        verbose=False,
    )
    pred = SAM3SemanticPredictor(overrides=overrides)
    _models[key] = pred
    return pred


def _get_sam_predictor(weights: Path, conf: float, device: str) -> Any:
    """SAM 2-style visual prompts (single box → one mask) on sam3.pt."""
    key = f"sam_{weights}_{conf}_{device}"
    if key in _models:
        return _models[key]
    from ultralytics import SAM

    model = SAM(str(weights))
    if device and device != "cpu":
        try:
            model.to(device)
        except Exception:
            pass
    _models[key] = (model, conf, device if device != "cpu" else "cpu")
    return _models[key]


def segment_with_text(
    image_bgr: np.ndarray,
    text_prompts: List[str],
    *,
    weights: Path | None = None,
    conf: float = 0.25,
    device: str = "cuda",
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Run SAM 3 concept segmentation for text class prompts.
    Returns detections (bbox, class_id=text index, score) and aligned masks.
    """
    import torch

    if not text_prompts:
        return [], []

    w = weights or resolve_sam3_weights()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    predictor = _get_semantic_predictor(w, conf, device)
    predictor.set_image(image_bgr)
    h, w_img = image_bgr.shape[:2]
    pred_masks, pred_boxes = predictor.inference_features(
        predictor.features,
        src_shape=(h, w_img),
        text=text_prompts,
    )
    return _tensors_to_detections_masks(pred_masks, pred_boxes, text_prompts)


def segment_with_yolo_boxes(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    *,
    weights: Path | None = None,
    conf: float = 0.25,
    device: str = "cuda",
    drop_unsegmentable: bool = False,
    min_mask_in_box_frac: float = 0.02,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Refine each YOLO/SAHI box with SAM 3 visual (box) prompt.
    Preserves YOLO class_id, score, and bbox (mask refines segmentation only).
    Optionally drops boxes where SAM 3 mask is empty inside the YOLO box.
    """
    from ultralytics import SAM

    if not detections:
        return [], []

    w = weights or resolve_sam3_weights()
    sam_model, _, dev = _get_sam_predictor(w, conf, device)
    refined: List[Dict[str, Any]] = []
    masks_out: List[np.ndarray] = []
    h, w_img = image_bgr.shape[:2]
    blank = np.zeros((h, w_img), dtype=np.uint8)
    boxes = [[float(v) for v in det["bbox"]] for det in detections]

    try:
        results = sam_model.predict(
            source=image_bgr,
            bboxes=boxes,
            conf=conf,
            device=dev,
            verbose=False,
        )
    except Exception:
        if drop_unsegmentable:
            return [], []
        return (
            [{**det, "sam3_refined": False} for det in detections],
            [blank.copy() for _ in detections],
        )

    r0 = results[0] if results else None
    mask_tensors = None
    if r0 is not None and r0.masks is not None and len(r0.masks):
        mask_tensors = r0.masks.data

    for i, det in enumerate(detections):
        if mask_tensors is None or i >= len(mask_tensors):
            if drop_unsegmentable:
                continue
            refined.append({**det, "sam3_refined": False})
            masks_out.append(blank.copy())
            continue
        mask = mask_tensors[i].cpu().numpy().astype(np.uint8)
        x1, y1, x2, y2 = det["bbox"]
        box_area = max(1.0, (float(x2) - float(x1)) * (float(y2) - float(y1)))
        mask_in_box = float(mask[int(y1):int(y2), int(x1):int(x2)].sum()) if mask.size else 0.0
        if mask.size == 0 or mask_in_box < min_mask_in_box_frac * box_area:
            if drop_unsegmentable:
                continue
            refined.append({**det, "sam3_refined": False})
            masks_out.append(blank.copy())
            continue
        # Keep original YOLO/SAHI bbox — tight mask boxes often shrink IoU vs GT.
        refined.append({**det, "sam3_refined": True})
        masks_out.append(mask)

    return refined, masks_out


def _tensors_to_detections_masks(
    pred_masks: Any,
    pred_boxes: Any,
    text_prompts: List[str],
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    detections: List[Dict[str, Any]] = []
    masks: List[np.ndarray] = []
    if pred_boxes is None or pred_masks is None:
        return detections, masks

    boxes = pred_boxes.cpu().numpy()
    mask_data = pred_masks.cpu().numpy()
    if boxes.shape[0] == 0:
        return detections, masks

    for i in range(len(boxes)):
        x1, y1, x2, y2, sc, cls = boxes[i]
        cid = int(cls)
        if cid < 0 or cid >= len(text_prompts):
            cid = min(max(cid, 0), len(text_prompts) - 1)
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "class_id": cid,
            "score": float(sc),
            "source": "sam3_text",
            "class_name": text_prompts[cid],
            "text_prompt": text_prompts[cid],
        })
        masks.append((mask_data[i] > 0.5).astype(np.uint8))

    return detections, masks
