#!/usr/bin/env python3
"""
Full supervisor benchmark: all 5 pipelines × all labeled test images × KITTI + VisDrone.

Outputs:
  outputs/supervisor_report/
    kitti_val/           per-mode metrics + supervisor_metrics.csv
    visdrone_test_dev/
    SUPERVISOR_REPORT.csv   combined (both datasets)
    SUPERVISOR_REPORT.md    human-readable summary

KITTI: uses images/val + labels/val (1496 images; no public labeled test split).
VisDrone: uses test-dev (1610 images; labels converted from annotations).

Usage:
  python scripts/run_supervisor_report.py --device cuda
  python scripts/run_supervisor_report.py --device cuda --limit 10   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.yolo_weights import resolve_yolo_checkpoint  # noqa: E402

OUT_ROOT = ROOT / "outputs" / "supervisor_report"
PYTHON = sys.executable

DATASETS = {
    "kitti_val": {
        "title": "KITTI validation (labeled eval split)",
        "images": ROOT / "kitti" / "images" / "val",
        "labels": ROOT / "kitti" / "labels" / "val",
        "weights_key": "kitti",
    },
    "visdrone_test_dev": {
        "title": "VisDrone test-dev",
        "images": ROOT
        / "dataset"
        / "VisDrone2019-DET-test-dev"
        / "VisDrone2019-DET-test-dev"
        / "images",
        "labels": ROOT
        / "dataset"
        / "VisDrone2019-DET-test-dev"
        / "VisDrone2019-DET-test-dev"
        / "labels",
        "weights_key": "visdrone",
    },
}


def _ensure_visdrone_test_labels() -> None:
    labels_dir = DATASETS["visdrone_test_dev"]["labels"]
    if labels_dir.is_dir() and any(labels_dir.glob("*.txt")):
        print(f"[INFO] VisDrone test-dev labels present: {labels_dir}")
        return
    print("[INFO] Converting VisDrone test-dev annotations → YOLO labels…")
    script = ROOT / "scripts" / "training" / "visdrone_to_yolo.py"
    subprocess.run(
        [PYTHON, str(script), "--split", "test-dev"],
        cwd=str(ROOT),
        check=True,
    )


def _run_dataset(
    key: str,
    cfg: dict,
    device: str,
    limit: int,
    out_root: Path,
    skip_existing: bool,
) -> Path:
    images = cfg["images"]
    labels = cfg["labels"]
    if not images.is_dir():
        raise FileNotFoundError(f"Missing images: {images}")
    if not labels.is_dir():
        raise FileNotFoundError(f"Missing labels: {labels}")

    weights = resolve_yolo_checkpoint(cfg["weights_key"], warn=True)
    out = out_root / key
    cmd = [
        PYTHON,
        str(ROOT / "pipelines" / "compare_all.py"),
        "--source", str(images),
        "--ground-truth", str(labels),
        "--weights", weights,
        "--output", str(out),
        "--device", device,
        "--skip-images",
        "--dataset-name", cfg["title"],
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if skip_existing:
        cmd.append("--skip-existing")

    print(f"\n{'='*70}\nBENCHMARK: {cfg['title']}\n{'='*70}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return out / "supervisor_metrics.csv"


def _merge_csvs(paths: list[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []
    header: list[str] | None = None
    for p in paths:
        if not p.is_file():
            print(f"[WARN] Missing {p}")
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            h = next(r, None)
            if h and header is None:
                header = h
            for row in r:
                rows.append(row)
    if not header:
        return
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_markdown(dest: Path, combined: Path) -> None:
    if not combined.is_file():
        return
    lines = [
        "# Supervisor Detection Metrics Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Datasets",
        "- **KITTI** — `kitti/images/val` + `kitti/labels/val` (no public labeled test split)",
        "- **VisDrone** — `VisDrone2019-DET-test-dev` (1610 images)",
        "",
        "## Pipelines",
        "1. yolo_only",
        "2. yolo_sahi",
        "3. sam3_only",
        "4. sam3_yolo",
        "5. yolo_sahi_sam3",
        "",
        f"Full table: `{combined.relative_to(ROOT).as_posix()}`",
        "",
        "## Results summary",
        "",
        "| Dataset | Pipeline | Precision | Recall | F1 | mAP50 | mAP50-95 | TP | FP | FN | Avg FPS | Total runtime (s) |",
        "|---------|----------|-----------|--------|-----|-------|----------|-----|-----|-----|---------|-------------------|",
    ]
    with combined.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            lines.append(
                f"| {row['dataset']} | {row['pipeline']} | {row['precision']} | {row['recall']} | "
                f"{row['f1']} | {row['map50']} | {row['map50_95']} | {row['true_positives']} | "
                f"{row['false_positives']} | {row['false_negatives']} | {row['avg_fps']} | "
                f"{row['total_runtime_s']} |"
            )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervisor full-dataset benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Max images per dataset (0=all)")
    parser.add_argument(
        "--datasets",
        default="kitti_val,visdrone_test_dev",
        help="Comma-separated keys",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Report output root (default: outputs/supervisor_report)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pipeline modes already benchmarked on disk",
    )
    args = parser.parse_args()

    out_root = Path(args.output_root) if args.output_root else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    _ensure_visdrone_test_labels()

    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    csv_paths: list[Path] = []
    for key in tqdm(keys, desc="Datasets", unit="dataset"):
        if key not in DATASETS:
            print(f"[ERROR] Unknown dataset key: {key}")
            sys.exit(1)
        csv_paths.append(
            _run_dataset(
                key, DATASETS[key], args.device, args.limit, out_root, args.skip_existing
            )
        )

    combined = out_root / "SUPERVISOR_REPORT.csv"
    _merge_csvs(csv_paths, combined)
    _write_markdown(out_root / "SUPERVISOR_REPORT.md", combined)
    print(f"\n[DONE] Combined report: {combined}")
    print(f"[DONE] Markdown: {out_root / 'SUPERVISOR_REPORT.md'}")


if __name__ == "__main__":
    main()
