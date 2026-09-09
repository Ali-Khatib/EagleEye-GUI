#!/usr/bin/env python3
"""Build comparison_report/VisDrone YOLOv8 report.txt from yolov8 benchmark folder."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "benchmark_working" / "supervisor_report" / "visdrone_test_dev_yolov8"
OUT = ROOT / "comparison_report"
REPORT = OUT / "VisDrone YOLOv8 report.txt"

MODES = [
    ("yolo_sahi_sam3", "yolo sahi sam3"),
]


def fmt(v, d=3):
    if v is None:
        return "n/a"
    return f"{float(v):.{d}f}"


def main() -> None:
    if not WORK.is_dir():
        print(f"[ERROR] Missing benchmark dir: {WORK}")
        sys.exit(1)

    rows = []
    for key, label in MODES:
        mp = WORK / key / "metrics.json"
        if not mp.is_file():
            rows.append((label, 0, None, "missing"))
            continue
        j = json.loads(mp.read_text(encoding="utf-8"))
        n = int(j.get("images_evaluated", 0))
        rows.append((label, n, j, "complete" if n >= 1610 else "partial"))

    lines = [
        "EagleEye AI Comparison Report (YOLOv8n backbone)",
        "Dataset: VisDrone test dev (1610 images)",
        f"Generated: {datetime.now().strftime('%Y %b %d  %H:%M')}",
        "Detector: runs/detect/train_v8_visdrone/weights/best.pt",
        "SAM3: frozen pretrained sam3.pt (same as YOLO11 benchmark)",
        "",
        "All pipelines",
        "",
        f"{'Pipeline':<18} {'Images':>7} {'Precision':>10} {'Recall':>8} {'F1':>8} "
        f"{'mAP50':>8} {'mAP50-95':>10} {'FPS':>7} {'Status':>8}",
        "-" * 95,
    ]
    for label, n, j, status in rows:
        if j is None:
            lines.append(f"{label:<18} {n:>7} {'n/a':>10} {'n/a':>8} {'n/a':>8} "
                         f"{'n/a':>8} {'n/a':>10} {'n/a':>7} {status:>8}")
        else:
            lines.append(
                f"{label:<18} {n:>7} {fmt(j.get('precision')):>10} {fmt(j.get('recall')):>8} "
                f"{fmt(j.get('f1')):>8} {fmt(j.get('map50')):>8} {fmt(j.get('map50_95')):>10} "
                f"{fmt(j.get('fps'), 2):>7} {status:>8}"
            )

    lines += [
        "",
        "Paper comparison note: compare yolo sahi and yolo sahi sam3 rows with SAHI+YOLOv8 results from the reference paper.",
        f"Per-pipeline details: comparison_report/VisDrone YOLOv8/<pipeline>/",
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    y8_dir = OUT / "VisDrone YOLOv8"
    y8_dir.mkdir(parents=True, exist_ok=True)
    for key, label in MODES:
        src = WORK / key
        dst = y8_dir / label
        if not (src / "metrics.json").is_file():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "results.csv", "supervisor_metrics.csv"):
            sp = src / name
            if sp.is_file():
                (dst / name).write_bytes(sp.read_bytes())

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {REPORT}")


if __name__ == "__main__":
    main()
