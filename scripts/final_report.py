#!/usr/bin/env python3
"""
Dataset-level pipeline report with detection-stage averages.

Reads compare_all mode output (metrics.json per mode) or aggregates in-memory
ModeAggregate objects.

Usage:
  python scripts/final_report.py --input outputs/compare/visdrone
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.compare_metrics import ModeAggregate  # noqa: E402


def _load_mode_metrics(mode_dir: Path) -> Dict[str, Any]:
    p = mode_dir / "metrics.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def build_report(aggregates: Dict[str, ModeAggregate | Dict[str, Any]]) -> str:
    lines: List[str] = [
        "EagleEye Pipeline Final Report",
        "=" * 60,
        "",
        "Detection metrics use final boxes only (YOLO + accepted SAHI + accepted SAM3 text).",
        "mask_count is separate and does not affect predicted_count.",
        "",
    ]
    for mode, agg in aggregates.items():
        if isinstance(agg, ModeAggregate):
            d = {
                "f1": agg.f1,
                "precision": agg.precision,
                "recall": agg.recall,
                "fps": agg.fps,
                "avg_raw_yolo_count": agg.avg_raw_yolo_count,
                "avg_filtered_yolo_count": agg.avg_filtered_yolo_count,
                "avg_raw_sahi_count": agg.avg_raw_sahi_count,
                "avg_accepted_sahi_count": agg.avg_accepted_sahi_count,
                "avg_raw_sam3_text_count": agg.avg_raw_sam3_text_count,
                "avg_accepted_sam3_text_count": agg.avg_accepted_sam3_text_count,
                "avg_mask_count": agg.avg_mask_count,
                "avg_final_prediction_count": agg.avg_final_prediction_count,
                "images_evaluated": agg.images_evaluated,
            }
        else:
            d = agg

        lines.extend([
            f"Mode: {mode}",
            f"  images: {d.get('images_evaluated', 'n/a')}",
            f"  F1={d.get('f1', 0):.3f}  P={d.get('precision', 0):.3f}  R={d.get('recall', 0):.3f}  FPS={d.get('fps', 0):.2f}",
            f"  avg_raw_yolo_count:          {d.get('avg_raw_yolo_count', 0):.2f}",
            f"  avg_filtered_yolo_count:     {d.get('avg_filtered_yolo_count', 0):.2f}",
            f"  avg_raw_sahi_count:          {d.get('avg_raw_sahi_count', 0):.2f}",
            f"  avg_accepted_sahi_count:     {d.get('avg_accepted_sahi_count', 0):.2f}",
            f"  avg_raw_sam3_text_count:     {d.get('avg_raw_sam3_text_count', 0):.2f}",
            f"  avg_accepted_sam3_text_count:{d.get('avg_accepted_sam3_text_count', 0):.2f}",
            f"  avg_mask_count:              {d.get('avg_mask_count', 0):.2f}",
            f"  avg_final_prediction_count:  {d.get('avg_final_prediction_count', 0):.2f}",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline final report with stage averages.")
    parser.add_argument("--input", required=True, help="compare_all output root")
    parser.add_argument("--output", default="", help="Report txt path (default: <input>/FINAL_REPORT.txt)")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}")
        sys.exit(1)

    modes = ("yolo_only", "yolo_sahi", "sam3_only", "sam3_yolo", "yolo_sahi_sam3")
    aggregates: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        md = _load_mode_metrics(root / mode)
        if md:
            aggregates[mode] = md

    if not aggregates:
        print(f"[ERROR] No metrics.json found under {root}")
        sys.exit(1)

    report = build_report(aggregates)
    out_path = Path(args.output) if args.output else root / "FINAL_REPORT.txt"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
