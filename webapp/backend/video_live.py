"""Live video MJPEG stream for the web UI (YOLO / YOLO+SAHI)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.compare_viz import draw_detections  # noqa: E402
from core.yolo_weights import resolve_yolo_checkpoint  # noqa: E402

DEMO_DIR = ROOT / "data" / "demo"
VIDEO_MODES = ("yolo_only", "yolo_sahi")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

router = APIRouter()
_yolo_cache: dict[str, object] = {}


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_video_path(dataset: str) -> Path:
    primary = DEMO_DIR / f"test_video_{dataset}.mp4"
    if primary.is_file():
        return primary
    for ext in VIDEO_EXTS:
        p = DEMO_DIR / f"test_video_{dataset}{ext}"
        if p.is_file():
            return p
    fallback = DEMO_DIR / "test2.mp4"
    return fallback if fallback.is_file() else primary


def _get_yolo(dataset: str):
    rel = resolve_yolo_checkpoint(dataset, warn=False)
    key = f"{dataset}:{rel}"
    if key not in _yolo_cache:
        from ultralytics import YOLO

        _yolo_cache[key] = YOLO(str(ROOT / rel))
    return _yolo_cache[key]


def _annotate(frame: np.ndarray, dataset: str, mode: str) -> np.ndarray:
    from pipelines import yolo_only, yolo_sahi

    device = _device()
    model = _get_yolo(dataset)
    names = getattr(model, "names", None) or {}
    if mode == "yolo_sahi":
        weights = resolve_yolo_checkpoint(dataset, warn=False)
        out = yolo_sahi.run(frame, weights, device)
    else:
        out = yolo_only.run(frame, model, device)
    vis = draw_detections(frame, out.detections, class_names=names, masks=out.masks)
    label = "YOLO + SAHI" if mode == "yolo_sahi" else "YOLO only"
    cv2.putText(
        vis, f"{label}  |  live  |  not SAM 3",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        vis, f"{label}  |  live  |  not SAM 3",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA,
    )
    return vis


def _mjpeg(dataset: str, mode: str) -> Iterator[bytes]:
    path = resolve_video_path(dataset)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="No video uploaded. Drop an mp4 on the Video tab.",
        )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Cannot open video: {path.name}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))
            vis = _annotate(frame, dataset, mode)
            ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if not ok:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
    finally:
        cap.release()


@router.get("/api/video")
def video_info(dataset: str = Query("visdrone")) -> dict:
    path = resolve_video_path(dataset)
    return {
        "ok": True,
        "dataset": dataset,
        "path": str(path) if path.is_file() else None,
        "ready": path.is_file(),
        "modes": list(VIDEO_MODES),
        "note": "Live stream is YOLO or YOLO+SAHI. Pipeline 5 (SAM 3) is image-only.",
    }


@router.post("/api/video/upload")
async def upload_video(
    dataset: str = Query("visdrone"),
    video: UploadFile = File(...),
) -> dict:
    name = video.filename or "clip.mp4"
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(status_code=400, detail="Use mp4, avi, mov, mkv, or webm.")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    dst = DEMO_DIR / f"test_video_{dataset}.mp4"
    data = await video.read()
    if len(data) < 1000:
        raise HTTPException(status_code=400, detail="Video file is empty.")
    dst.write_bytes(data)
    return {"ok": True, "dataset": dataset, "path": str(dst), "bytes": len(data)}


@router.get("/api/video/stream")
def stream_video(
    dataset: str = Query("visdrone"),
    mode: str = Query("yolo_only"),
):
    m = mode.lower().strip()
    if m not in VIDEO_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {VIDEO_MODES}")
    return StreamingResponse(
        _mjpeg(dataset, m),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store"},
    )
