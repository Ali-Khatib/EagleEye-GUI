"""
FastAPI backend for the SAHI / YOLO / SAM pipeline comparison UI.

Each endpoint accepts a `dataset` query parameter (or form field for upload):
    dataset = "visdrone"  (default)
    dataset = "kitti"

Per-dataset state means you can run a pipeline on VisDrone, then switch to
KITTI and run again, and both results are kept separate on disk:
    outputs/webapp/visdrone/yolo only/result.jpg
    outputs/webapp/kitti/yolo sahi/metrics.csv

Run:
    cd webapp/backend
    pip install -r requirements.txt
    python main.py

Project: TUBITAK ARDEB 3501 - 124E099
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from video_live import router as video_router


# -----------------------------------------------------------------------------
# Paths / datasets
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]            # SAHI/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import YOLO_VISDRONE_V8  # noqa: E402
from core.webapp_outputs import (  # noqa: E402
    PIPELINE_FOLDERS,
    input_image_path,
    metrics_csv_path,
    pipeline_dir,
    result_image_path,
    webapp_root,
)
from core.yolo_weights import yolo_checkpoint_map  # noqa: E402

RESULTS_DIR = webapp_root(ROOT)
DEMO_DIR = ROOT / "data" / "demo"
EXPERIMENTS_DIR = ROOT / "experiments"

DATASETS = ("visdrone", "kitti", "stock")
DEFAULT_DATASET = "visdrone"

# Fine-tuned YOLO weights (VisDrone train9 = 100 epochs, KITTI train2).
YOLO_CHECKPOINTS: Dict[str, str] = yolo_checkpoint_map()

# SAM 3 weights (Ultralytics sam3.pt from Hugging Face facebook/sam3).
SAM3_DEFAULT = "sam3.pt"

DATASET_LABEL: Dict[str, str] = {
    "visdrone": "VisDrone",
    "kitti": "KITTI",
    "stock": "Stock (COCO)",
}


def _dataset_yolo_path(dataset: str) -> Path:
    return ROOT / YOLO_CHECKPOINTS.get(dataset, "")


def _dataset_yolo_ready(dataset: str) -> bool:
    return _dataset_yolo_path(dataset).is_file()


def _yolov8_visdrone_ready() -> bool:
    return YOLO_VISDRONE_V8.is_file()


def _experiment_weights_ready(e: Dict[str, Any], dataset: str) -> bool:
    if e.get("needs_yolov8"):
        return dataset == "visdrone" and _yolov8_visdrone_ready()
    if "YOLO" in e.get("components", []):
        return _dataset_yolo_ready(dataset)
    return True


def _dataset_sam_path(dataset: str) -> Path:
    try:
        from core.sam3_support import resolve_sam3_weights
        return resolve_sam3_weights()
    except FileNotFoundError:
        return ROOT / SAM3_DEFAULT


def _dataset_sam_ready(dataset: str) -> bool:
    return _dataset_sam_path(dataset).is_file()


def _check_dataset(dataset: str) -> str:
    d = dataset.lower().strip()
    if d not in DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset '{dataset}'. Use one of: {DATASETS}",
        )
    return d


def _image_path(dataset: str) -> Path:
    """Per-dataset image path with fallback to legacy demo_data/test_image.jpg."""
    primary = DEMO_DIR / f"test_image_{dataset}.jpg"
    if primary.exists():
        return primary
    fallback = DEMO_DIR / "test_image.jpg"
    return fallback if fallback.exists() else primary


def _label_path(dataset: str) -> Path:
    primary = DEMO_DIR / f"test_image_{dataset}.txt"
    if primary.exists():
        return primary
    fallback = DEMO_DIR / "test_image.txt"
    return fallback if fallback.exists() else primary


# -----------------------------------------------------------------------------
# Dataset val-set roots (used by /api/dataset/random-sample)
#
# VisDrone validation set:
#   images:      dataset/VisDrone2019-DET-val/VisDrone2019-DET-val/images/*.jpg
#   annotations: dataset/VisDrone2019-DET-val/VisDrone2019-DET-val/annotations/*.txt
#                (VisDrone CSV format — needs conversion to YOLO)
#
# KITTI validation set (already YOLO format):
#   images: kitti/images/val/*.png
#   labels: kitti/labels/val/*.txt
# -----------------------------------------------------------------------------
DATASET_SAMPLE_DIRS: Dict[str, Dict[str, Any]] = {
    "visdrone": {
        "images": ROOT / "dataset" / "VisDrone2019-DET-val"
                  / "VisDrone2019-DET-val" / "images",
        "labels": ROOT / "dataset" / "VisDrone2019-DET-val"
                  / "VisDrone2019-DET-val" / "annotations",
        "label_format": "visdrone",
        "image_glob": "*.jpg",
    },
    "kitti": {
        "images": ROOT / "kitti" / "images" / "val",
        "labels": ROOT / "kitti" / "labels" / "val",
        "label_format": "yolo",
        "image_glob": "*.png",
    },
    # "stock" reuses VisDrone val images so the COCO-pretrained YOLO has
    # something realistic to look at. The GT label is converted from VisDrone
    # format to YOLO so class-agnostic IoU still works.
    "stock": {
        "images": ROOT / "dataset" / "VisDrone2019-DET-val"
                  / "VisDrone2019-DET-val" / "images",
        "labels": ROOT / "dataset" / "VisDrone2019-DET-val"
                  / "VisDrone2019-DET-val" / "annotations",
        "label_format": "visdrone",
        "image_glob": "*.jpg",
    },
}


def _visdrone_to_yolo(
    visdrone_txt: Path, img_w: int, img_h: int
) -> List[str]:
    """
    Convert a VisDrone annotation file to YOLO normalized format.
    VisDrone row: bbox_left, bbox_top, bbox_width, bbox_height,
                  score, object_category, truncation, occlusion
    object_category: 0=ignored regions, 1..10 are real classes,
                     11=others (skipped).
    YOLO class id = visdrone_category - 1  (matches VisDrone.yaml: 0..9).
    """
    lines: List[str] = []
    if not visdrone_txt.exists():
        return lines
    with open(visdrone_txt, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            row = [r.strip() for r in raw.strip().split(",")]
            if len(row) < 6:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
                w = float(row[2])
                h = float(row[3])
                cat = int(row[5])
            except ValueError:
                continue
            if cat <= 0 or cat > 10:        # ignored / others
                continue
            cls = cat - 1
            xc = (x + w / 2.0) / img_w
            yc = (y + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h
            if nw <= 0 or nh <= 0:
                continue
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _img_size(path: Path) -> Tuple[int, int]:
    """Return (width, height). Uses cv2 if available; else fails."""
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"cv2 could not read {path}")
        h, w = img.shape[:2]
        return int(w), int(h)
    except Exception as ex:
        raise HTTPException(status_code=500,
                            detail=f"Could not read image size: {ex}")


def _save_image_jpg_copy(src: Path, dst: Path) -> int:
    """
    Copy an image to dst, re-encoding as JPEG when needed (e.g. KITTI .png).
    Returns the resulting file size in bytes.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in (".jpg", ".jpeg"):
        shutil.copyfile(src, dst)
        return dst.stat().st_size
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(src))
        if img is None:
            raise RuntimeError("cv2 could not decode source image")
        cv2.imwrite(str(dst), img)
        return dst.stat().st_size
    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Could not convert image to JPEG: {ex}",
        )


# -----------------------------------------------------------------------------
# Experiments registry
# -----------------------------------------------------------------------------
EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "id": "1",
        "key": "yolo_only",
        "name": "YOLO only",
        "subtitle": "Single-pass full-frame object detection.",
        "description": (
            "Standard YOLO inference on the full-resolution image. "
            "Fast baseline; small or distant objects may be missed."
        ),
        "components": ["YOLO"],
        "script": "exp1_yolo_only.py",
        "basename": "01_yolo_only",
        "pipeline_folder": "yolo only",
        "metric_kind": "class_aware",
        "needs_sam3": False,
    },
    {
        "id": "2",
        "key": "yolo_sahi",
        "name": "YOLO + SAHI",
        "subtitle": "Sliced inference for small / crowded objects.",
        "description": (
            "SAHI tiles the image, runs YOLO on each slice, merges with "
            "GREEDYNMM and extra NMS. Tuned for small-object recall."
        ),
        "components": ["SAHI", "YOLO"],
        "script": "exp2_yolo_sahi.py",
        "basename": "02_yolo_sahi",
        "pipeline_folder": "yolo sahi",
        "metric_kind": "class_aware",
        "needs_sam3": False,
    },
    {
        "id": "3",
        "key": "sam3_only",
        "name": "SAM 3 only",
        "subtitle": "Text-prompt open-vocabulary segmentation.",
        "description": (
            "SAM 3 concept segmentation with text prompts "
            "(person, car, truck, …). Class-agnostic box metrics unless "
            "prompts align with YOLO ground-truth class IDs."
        ),
        "components": ["SAM3"],
        "script": "exp3_sam3_only.py",
        "basename": "03_sam3_only",
        "pipeline_folder": "sam3 only",
        "metric_kind": "class_agnostic",
        "needs_sam3": True,
    },
    {
        "id": "4",
        "key": "sam3_yolo",
        "name": "SAM 3 + YOLO",
        "subtitle": "YOLO detects; SAM 3 refines masks per box.",
        "description": (
            "YOLO proposes boxes and class IDs; SAM 3 refines a mask inside "
            "each box only—no extra unlabeled SAM detections."
        ),
        "components": ["YOLO", "SAM3"],
        "script": "exp4_sam3_yolo.py",
        "basename": "04_sam3_yolo",
        "pipeline_folder": "sam3 yolo",
        "metric_kind": "class_aware",
        "needs_sam3": True,
    },
    {
        "id": "5",
        "key": "yolo_sahi_sam3",
        "name": "YOLO + SAHI + SAM 3",
        "subtitle": "Main pipeline: detect small objects, refine masks.",
        "description": (
            "SAHI + YOLO for detection on slices, merge + dedupe, then SAM 3 "
            "mask refinement per detection. Best chance on small, distant, "
            "and crowded scenes while preserving YOLO class labels."
        ),
        "components": ["SAHI", "YOLO", "SAM3"],
        "script": "exp5_yolo_sahi_sam3.py",
        "basename": "05_yolo_sahi_sam3",
        "pipeline_folder": "yolo sahi sam3",
        "metric_kind": "class_aware",
        "needs_sam3": True,
    },
    {
        "id": "6",
        "key": "yolov8_sahi_sam3",
        "name": "YOLOv8 + SAHI + SAM 3",
        "subtitle": "Same pipeline as #5, YOLOv8n backbone (VisDrone paper compare).",
        "description": (
            "Identical SAHI + SAM 3 stack as the main pipeline, but uses the "
            "fine-tuned YOLOv8n checkpoint (train_v8_visdrone) instead of YOLO11. "
            "VisDrone only — for comparing with published SAHI + YOLOv8 results."
        ),
        "components": ["SAHI", "YOLOv8", "SAM3"],
        "script": "exp6_yolov8_sahi_sam3.py",
        "basename": "06_yolov8_sahi_sam3",
        "pipeline_folder": "yolov8 sahi sam3",
        "metric_kind": "class_aware",
        "needs_sam3": True,
        "needs_yolov8": True,
        "yolov8_path": "runs/detect/train_v8_visdrone/weights/best.pt",
        "visdrone_only": True,
    },
]


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="EagleEye AI API", version="2.0.0")
app.include_router(video_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _empty_run() -> Dict[str, Any]:
    return {"status": "idle", "stdout": "", "stderr": "", "elapsed": 0.0, "started_at": 0.0}


def _run_elapsed(state: Dict[str, Any]) -> float:
    if state.get("status") == "running" and state.get("started_at"):
        return round(time.time() - state["started_at"], 1)
    return float(state.get("elapsed") or 0.0)


SAM3_TIMEOUT_SEC = 1800
DEFAULT_TIMEOUT_SEC = 900


# RUN_STATE[dataset][exp_id]
RUN_STATE: Dict[str, Dict[str, Dict[str, Any]]] = {
    d: {e["id"]: _empty_run() for e in EXPERIMENTS} for d in DATASETS
}


def _get_experiment(exp_id: str) -> Dict[str, Any]:
    for e in EXPERIMENTS:
        if e["id"] == exp_id:
            return e
    raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")


def _basename_for(e: Dict[str, Any], dataset: str) -> str:
    """API id: dataset/pipeline folder (legacy field name basename)."""
    folder = e.get("pipeline_folder") or PIPELINE_FOLDERS.get(e["basename"], e["basename"])
    return f"{dataset}/{folder}"


def _pipeline_output_dir(e: Dict[str, Any], dataset: str) -> Path:
    folder = e.get("pipeline_folder") or PIPELINE_FOLDERS.get(e["basename"], e["basename"])
    return RESULTS_DIR / dataset / folder


def _read_metrics(basename: str) -> Optional[Dict[str, Any]]:
    # basename is "dataset/pipeline folder"
    parts = basename.split("/", 1)
    if len(parts) != 2:
        return None
    csv_path = RESULTS_DIR / parts[0] / parts[1] / "metrics.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return next(reader, None)
    except Exception:
        return None


def _has_result_image(basename: str) -> bool:
    parts = basename.split("/", 1)
    if len(parts) != 2:
        return False
    return (RESULTS_DIR / parts[0] / parts[1] / "result.jpg").exists()


def _has_pipeline_output(basename: str) -> bool:
    """True when a web run produced at least a result image or metrics file."""
    if _has_result_image(basename):
        return True
    parts = basename.split("/", 1)
    if len(parts) != 2:
        return False
    return (RESULTS_DIR / parts[0] / parts[1] / "metrics.csv").is_file()


def _inference_device_info() -> Dict[str, Any]:
    """CUDA status for the web UI (pipelines prefer GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "inference_device": "cuda",
                "cuda_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
            }
    except Exception:
        pass
    return {
        "inference_device": "cpu",
        "cuda_available": False,
        "gpu_name": None,
    }


def _subprocess_env() -> Dict[str, str]:
    """Pipeline subprocess env — always prefer CUDA when available."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    info = _inference_device_info()
    env["EXP_DEVICE"] = "cuda" if info["cuda_available"] else "cpu"
    return env
@app.get("/api/health")
def health(dataset: str = Query(DEFAULT_DATASET)) -> Dict[str, Any]:
    d = _check_dataset(dataset)
    img = _image_path(d)
    lbl = _label_path(d)
    yolo = _dataset_yolo_path(d)
    sam = _dataset_sam_path(d)
    return {
        "status": "ok",
        "datasets": list(DATASETS),
        "dataset": d,
        "root": str(ROOT),
        "results_dir": str(RESULTS_DIR),
        "results_dir_exists": RESULTS_DIR.exists(),
        "image": str(img),
        "image_exists": img.exists(),
        "label": str(lbl),
        "label_exists": lbl.exists(),
        "yolo_path": str(yolo),
        "yolo_ready": yolo.is_file(),
        "sam_path": str(sam),
        "sam_ready": sam.is_file(),
        **_inference_device_info(),
    }


@app.get("/api/datasets")
def list_datasets() -> List[Dict[str, Any]]:
    """
    Return one record per dataset with its trained-YOLO availability so the
    UI can disable the toggle for datasets whose checkpoint isn't ready yet
    (e.g. KITTI while training is still running).
    """
    out = []
    for d in DATASETS:
        yolo = _dataset_yolo_path(d)
        sam = _dataset_sam_path(d)
        out.append({
            "id": d,
            "label": DATASET_LABEL.get(d, d),
            "yolo_path": str(yolo),
            "yolo_ready": yolo.is_file(),
            "sam_path": str(sam),
            "sam_ready": sam.is_file(),
            "image_path": str(_image_path(d)),
            "image_ready": _image_path(d).exists(),
            "label_path": str(_label_path(d)),
            "label_ready": _label_path(d).exists(),
        })
    return out


@app.get("/api/experiments")
def list_experiments(
    dataset: str = Query(DEFAULT_DATASET),
) -> List[Dict[str, Any]]:
    d = _check_dataset(dataset)
    out = []
    for e in EXPERIMENTS:
        bn = _basename_for(e, d)
        out.append({
            **e,
            "dataset": d,
            "basename": bn,
            "has_result": _has_pipeline_output(bn),
            "metrics": _read_metrics(bn),
            "run_status": RUN_STATE[d][e["id"]]["status"],
            "elapsed": _run_elapsed(RUN_STATE[d][e["id"]]),
            "weights_ready": _experiment_weights_ready(e, d),
        })
    return out


def _run_experiment_worker(
    dataset: str,
    exp_id: str,
    module: str,
    state: Dict[str, Any],
) -> None:
    t0 = time.time()
    timeout = SAM3_TIMEOUT_SEC if exp_id in ("3", "4", "5", "6") else DEFAULT_TIMEOUT_SEC
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module, dataset],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_subprocess_env(),
        )
        elapsed = time.time() - t0
        state["stdout"] = proc.stdout or ""
        state["stderr"] = proc.stderr or ""
        state["elapsed"] = round(elapsed, 3)
        state["status"] = "done" if proc.returncode == 0 else "error"
        if proc.returncode != 0 and not state["stderr"]:
            state["stderr"] = f"Process exited with code {proc.returncode}"
    except subprocess.TimeoutExpired as ex:
        state["status"] = "error"
        state["stderr"] = f"Timeout after {timeout}s: {ex}"
        state["elapsed"] = round(time.time() - t0, 3)
    except Exception as ex:
        state["status"] = "error"
        state["stderr"] = str(ex)
        state["elapsed"] = round(time.time() - t0, 3)


@app.get("/api/results")
def list_results(
    dataset: str = Query(DEFAULT_DATASET),
) -> List[Dict[str, Any]]:
    d = _check_dataset(dataset)
    out = []
    for e in EXPERIMENTS:
        bn = _basename_for(e, d)
        out.append({
            **e,
            "dataset": d,
            "basename": bn,
            "has_result": _has_pipeline_output(bn),
            "metrics": _read_metrics(bn),
            "run_status": RUN_STATE[d][e["id"]]["status"],
            "elapsed": _run_elapsed(RUN_STATE[d][e["id"]]),
            "weights_ready": _experiment_weights_ready(e, d),
        })
    return out


@app.get("/api/results/{exp_id}")
def get_result(
    exp_id: str,
    dataset: str = Query(DEFAULT_DATASET),
) -> Dict[str, Any]:
    d = _check_dataset(dataset)
    e = _get_experiment(exp_id)
    bn = _basename_for(e, d)
    return {
        **e,
        "dataset": d,
        "basename": bn,
        "has_result": _has_pipeline_output(bn),
        "metrics": _read_metrics(bn),
        "run_status": RUN_STATE[d][exp_id]["status"],
        "stdout": RUN_STATE[d][exp_id]["stdout"],
        "stderr": RUN_STATE[d][exp_id]["stderr"],
        "elapsed": _run_elapsed(RUN_STATE[d][exp_id]),
    }


@app.post("/api/experiments/{exp_id}/run")
def run_experiment(
    exp_id: str,
    dataset: str = Query(DEFAULT_DATASET),
) -> Dict[str, Any]:
    d = _check_dataset(dataset)
    e = _get_experiment(exp_id)
    script_path = EXPERIMENTS_DIR / e["script"]
    if not script_path.exists():
        raise HTTPException(status_code=400,
                            detail=f"Script not found: {script_path}")

    if e.get("needs_sam3") and not _dataset_sam_ready(d):
        raise HTTPException(
            status_code=400,
            detail=(
                "SAM 3 weights not found. Place sam3.pt at the repo root or "
                "set SAM3_WEIGHTS."
            ),
        )

    if e.get("needs_yolov8"):
        if d != "visdrone":
            raise HTTPException(
                status_code=400,
                detail="YOLOv8 + SAHI + SAM 3 is available on VisDrone only.",
            )
        if not _yolov8_visdrone_ready():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"YOLOv8 weights not found at {YOLO_VISDRONE_V8}. "
                    "Train with scripts/training/train_yolo_visdrone_v8.py first."
                ),
            )

    if "YOLO" in e.get("components", []) and not e.get("needs_yolov8"):
        if not _dataset_yolo_ready(d):
            raise HTTPException(
                status_code=400,
                detail=f"YOLO checkpoint not ready for dataset '{d}'.",
            )

    state = RUN_STATE[d][exp_id]
    if state["status"] == "running":
        return {
            "ok": False,
            "exp_id": exp_id,
            "dataset": d,
            "status": "running",
            "elapsed": _run_elapsed(state),
            "stdout": state["stdout"],
            "stderr": "Pipeline already running.",
            "metrics": _read_metrics(_basename_for(e, d)),
        }

    state["status"] = "running"
    state["stdout"] = ""
    state["stderr"] = ""
    state["elapsed"] = 0.0
    state["started_at"] = time.time()

    module = f"experiments.{script_path.stem}"
    thread = threading.Thread(
        target=_run_experiment_worker,
        args=(d, exp_id, module, state),
        daemon=True,
    )
    thread.start()

    return {
        "ok": True,
        "exp_id": exp_id,
        "dataset": d,
        "status": "running",
        "elapsed": 0.0,
        "stdout": "",
        "stderr": "",
        "metrics": _read_metrics(_basename_for(e, d)),
    }


@app.get("/api/image/original")
def get_original_image(dataset: str = Query(DEFAULT_DATASET)):
    d = _check_dataset(dataset)
    webapp_img = input_image_path(ROOT, d)
    if webapp_img.is_file():
        return FileResponse(str(webapp_img), media_type="image/jpeg")
    img = _image_path(d)
    if not img.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {img}")
    return FileResponse(str(img), media_type="image/jpeg")


@app.get("/api/image/result/{exp_id}")
def get_result_image(
    exp_id: str,
    dataset: str = Query(DEFAULT_DATASET),
):
    d = _check_dataset(dataset)
    e = _get_experiment(exp_id)
    bn = _basename_for(e, d)
    img = _pipeline_output_dir(e, d) / "result.jpg"
    if not img.exists():
        raise HTTPException(status_code=404,
                            detail=f"Result image not found: {img}")
    return FileResponse(str(img), media_type="image/jpeg")


@app.get("/api/run-state")
def all_run_state() -> Dict[str, Any]:
    return RUN_STATE


# -----------------------------------------------------------------------------
# Upload + clear results
# -----------------------------------------------------------------------------
def _save_image_as_jpg(content: bytes, target: Path) -> None:
    """Re-encode bytes as JPEG. Fall back to raw bytes if cv2 isn't installed."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return

    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400,
                            detail="Could not decode uploaded image.")
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), img)


def _clear_results_dir(dataset: Optional[str] = None) -> int:
    """Delete webapp pipeline outputs for one dataset or all."""
    if not RESULTS_DIR.exists():
        return 0
    n = 0
    datasets = [dataset] if dataset else list(DATASETS)
    for ds in datasets:
        if ds is None:
            continue
        ds_path = RESULTS_DIR / ds
        if not ds_path.is_dir():
            continue
        inp = ds_path / "input_image.jpg"
        if inp.is_file():
            try:
                inp.unlink()
                n += 1
            except Exception:
                pass
        for sub in ds_path.iterdir():
            if not sub.is_dir():
                continue
            for fname in ("result.jpg", "metrics.csv", "notes.txt"):
                f = sub / fname
                if f.is_file():
                    try:
                        f.unlink()
                        n += 1
                    except Exception:
                        pass

    if dataset is None:
        for d in DATASETS:
            for k in RUN_STATE[d]:
                RUN_STATE[d][k] = _empty_run()
    else:
        for k in RUN_STATE[dataset]:
            RUN_STATE[dataset][k] = _empty_run()
    return n


@app.post("/api/upload")
async def upload_image(
    image: UploadFile = File(...),
    label: Optional[UploadFile] = File(None),
    dataset: str = Form(DEFAULT_DATASET),
    clear_results: bool = Form(True),
) -> Dict[str, Any]:
    """
    Replace the active image (and optional ground-truth label) for `dataset`.
    Files are saved to:
        data/demo/test_image_<dataset>.jpg
        data/demo/test_image_<dataset>.txt   (optional)

    By default also clears existing result files for the same dataset so the
    leaderboard reflects the new image.
    """
    d = _check_dataset(dataset)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    target_image = DEMO_DIR / f"test_image_{d}.jpg"
    target_label = DEMO_DIR / f"test_image_{d}.txt"

    if image is None:
        raise HTTPException(status_code=400, detail="No image file provided.")
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    _save_image_as_jpg(img_bytes, target_image)

    label_saved = False
    if label is not None and label.filename:
        label_bytes = await label.read()
        if label_bytes:
            target_label.write_bytes(label_bytes)
            label_saved = True
    elif target_label.is_file():
        # Custom image without label — remove stale GT from a previous image.
        try:
            target_label.unlink()
        except Exception:
            pass

    cleared = _clear_results_dir(d) if clear_results else 0

    return {
        "ok": True,
        "dataset": d,
        "image_path": str(target_image),
        "image_size_bytes": target_image.stat().st_size,
        "label_saved": label_saved,
        "label_path": str(target_label) if label_saved else None,
        "has_ground_truth": label_saved,
        "results_cleared": cleared,
    }


@app.post("/api/dataset/random-sample")
def random_sample(
    dataset: str = Query(DEFAULT_DATASET),
    clear_results: bool = Query(True),
    seed: Optional[int] = Query(None),
) -> Dict[str, Any]:
    """
    Pick a random validation image (with its ground-truth label) from the
    bundled dataset and copy it into data/demo/test_image_<dataset>.jpg/.txt.

    For VisDrone the annotation is converted from VisDrone CSV format to
    YOLO normalized format on the fly. For KITTI the label is already YOLO.
    """
    d = _check_dataset(dataset)
    cfg = DATASET_SAMPLE_DIRS.get(d)
    if cfg is None:
        raise HTTPException(status_code=400,
                            detail=f"No sample config for dataset '{d}'.")

    img_dir: Path = cfg["images"]
    lbl_dir: Path = cfg["labels"]
    fmt: str = cfg["label_format"]
    glob: str = cfg["image_glob"]

    if not img_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Validation images folder not found: {img_dir}",
        )
    candidates = sorted(img_dir.glob(glob))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No '{glob}' images found in {img_dir}",
        )

    rng = random.Random(seed) if seed is not None else random
    src_img = rng.choice(candidates)
    src_lbl = lbl_dir / (src_img.stem + ".txt")

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    dst_img = DEMO_DIR / f"test_image_{d}.jpg"
    dst_lbl = DEMO_DIR / f"test_image_{d}.txt"

    image_bytes = _save_image_jpg_copy(src_img, dst_img)

    label_saved = False
    label_count = 0
    if src_lbl.exists():
        if fmt == "yolo":
            shutil.copyfile(src_lbl, dst_lbl)
            try:
                with open(dst_lbl, "r", encoding="utf-8") as f:
                    label_count = sum(
                        1 for line in f if line.strip()
                    )
            except Exception:
                label_count = 0
            label_saved = True
        elif fmt == "visdrone":
            w, h = _img_size(dst_img)
            yolo_lines = _visdrone_to_yolo(src_lbl, w, h)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            label_count = len(yolo_lines)
            label_saved = True
    else:
        if dst_lbl.exists():
            try:
                dst_lbl.unlink()
            except Exception:
                pass

    cleared = _clear_results_dir(d) if clear_results else 0

    return {
        "ok": True,
        "dataset": d,
        "source_image": str(src_img),
        "source_label": str(src_lbl) if src_lbl.exists() else None,
        "image_path": str(dst_img),
        "image_size_bytes": image_bytes,
        "label_saved": label_saved,
        "label_path": str(dst_lbl) if label_saved else None,
        "label_count": label_count,
        "label_format": fmt,
        "results_cleared": cleared,
    }


@app.delete("/api/label")
def delete_label(dataset: str = Query(DEFAULT_DATASET)) -> Dict[str, Any]:
    d = _check_dataset(dataset)
    target = DEMO_DIR / f"test_image_{d}.txt"
    if target.exists():
        target.unlink()
        return {"ok": True, "dataset": d, "removed": True}
    return {"ok": True, "dataset": d, "removed": False}


@app.delete("/api/results")
def clear_results(
    dataset: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """If dataset is given clear only that dataset's results, else clear all."""
    if dataset is not None:
        d = _check_dataset(dataset)
        n = _clear_results_dir(d)
        return {"ok": True, "dataset": d, "cleared": n}
    n = _clear_results_dir(None)
    return {"ok": True, "dataset": None, "cleared": n}


# -----------------------------------------------------------------------------
# Local dev entry-point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
