#!/usr/bin/env python3
"""Copy Meta SAM3 training checkpoint into runs/sam3_finetune/{dataset}/sam3.pt for inference."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["visdrone", "kitti"], required=True)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to checkpoint.pt (default: runs/sam3_finetune/{dataset}/checkpoints/checkpoint.pt)",
    )
    args = parser.parse_args()

    out_dir = ROOT / "runs" / "sam3_finetune" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "sam3.pt"

    if args.checkpoint:
        src = Path(args.checkpoint)
    else:
        src = out_dir / "checkpoints" / "checkpoint.pt"

    if not src.is_file():
        print(f"[ERROR] Checkpoint not found: {src}")
        sys.exit(1)

    shutil.copy2(src, dest)
    mb = dest.stat().st_size / (1024 * 1024)
    print(f"[OK] Fine-tuned weights ready for pipelines: {dest} ({mb:.1f} MB)")
    print("Re-run SAM3 pipelines — compare_all will auto-pick this file per dataset.")


if __name__ == "__main__":
    main()
