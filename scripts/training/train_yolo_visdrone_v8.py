"""Fine-tune YOLOv8n on VisDrone (same schedule as YOLO11 train9)."""
from __future__ import annotations

import argparse
import sys
from multiprocessing import freeze_support
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n on VisDrone")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in --name run folder")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="train_v8_visdrone")
    parser.add_argument(
        "--benchmark-after",
        action="store_true",
        help="Run VisDrone test-dev benchmark when training finishes",
    )
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    data_yaml = "dataset/VisDrone.yaml"
    run_weights = PROJECT_ROOT / "runs" / "detect" / args.name / "weights"
    if args.resume:
        last_pt = run_weights / "last.pt"
        if not last_pt.is_file():
            raise FileNotFoundError(f"Cannot resume: missing {last_pt}")
        print(f"[INFO] Resuming from {last_pt} -> {args.epochs} epochs total")
        model = YOLO(str(last_pt))
        model.train(
            resume=True,
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=2,
            cache=False,
            amp=True,
            verbose=True,
        )
    else:
        model = YOLO("yolov8n.pt")
        model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=2,
            cache=False,
            amp=True,
            pretrained=True,
            verbose=True,
            name=args.name,
        )
    best = PROJECT_ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print(f"[DONE] Best weights: {best}")

    if args.benchmark_after and best.is_file():
        import subprocess

        script = PROJECT_ROOT / "scripts" / "run_visdrone_yolov8_benchmark.ps1"
        print(f"[INFO] Starting post-train benchmark via {script}")
        subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            cwd=str(PROJECT_ROOT),
            check=True,
        )


if __name__ == "__main__":
    freeze_support()
    main()
