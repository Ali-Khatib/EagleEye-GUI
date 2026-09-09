"""Convert VisDrone-format annotations to YOLO labels in place.

The official Ultralytics VisDrone YAML downloads the dataset and runs an
embedded converter to create `labels/` next to `images/`. Because we already
have the dataset at `dataset/VisDrone2019-DET-<split>/VisDrone2019-DET-<split>/`
this script reproduces that converter without re-downloading anything.

VisDrone annotation row format (CSV):
    bbox_left, bbox_top, bbox_width, bbox_height,
    score, object_category, truncation, occlusion

Mapping rules (matches the Ultralytics ship-with converter exactly so metrics
line up with the trained YOLO11n in `runs/detect/train9`):
    - skip rows where score == 0  (VisDrone "ignored regions")
    - YOLO class id = object_category - 1
    - drop rows where the resulting class id is outside 0..9
      (VisDrone has an extra "others" category that the trained model does
      not know about)
    - YOLO box: normalized x_center, y_center, width, height
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
    "test-challenge": "VisDrone2019-DET-test-challenge",
}

NUM_YOLO_CLASSES = 10  # pedestrian..motor (matches dataset/VisDrone.yaml)


def _convert_box(img_w: int, img_h: int, box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    return (
        (x + w / 2) * dw,
        (y + h / 2) * dh,
        w * dw,
        h * dh,
    )


def convert_split(split: str, overwrite: bool = False) -> int:
    if split not in SPLITS:
        raise SystemExit(f"unknown split '{split}'. choose from: {list(SPLITS)}")

    base = DATASET_ROOT / SPLITS[split] / SPLITS[split]
    images_dir = base / "images"
    annotations_dir = base / "annotations"
    labels_dir = base / "labels"

    if not images_dir.is_dir():
        raise SystemExit(f"missing images folder: {images_dir}")
    if not annotations_dir.is_dir():
        raise SystemExit(f"missing annotations folder: {annotations_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(annotations_dir.glob("*.txt"))
    if not ann_files:
        raise SystemExit(f"no annotation .txt files in {annotations_dir}")

    converted = 0
    skipped_no_image = 0
    skipped_existing = 0

    for ann_path in tqdm(ann_files, desc=f"converting {split}"):
        out_path = labels_dir / ann_path.name
        if out_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        img_path = images_dir / (ann_path.stem + ".jpg")
        if not img_path.is_file():
            skipped_no_image += 1
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        out_lines: list[str] = []
        with ann_path.open("r", encoding="utf-8") as f:
            for raw in f.read().strip().splitlines():
                row = raw.split(",")
                if len(row) < 6:
                    continue
                # Ultralytics' shipped rule: skip rows where score == 0.
                if row[4] == "0":
                    continue
                try:
                    cls = int(row[5]) - 1
                    box = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                except ValueError:
                    continue
                if cls < 0 or cls >= NUM_YOLO_CLASSES:
                    continue
                if box[2] <= 0 or box[3] <= 0:
                    continue
                xc, yc, w, h = _convert_box(img_w, img_h, box)
                out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        with out_path.open("w", encoding="utf-8") as fl:
            fl.writelines(out_lines)
        converted += 1

    print(
        f"[{split}] wrote {converted} label files into {labels_dir}"
        + (f" (skipped {skipped_existing} already-present)" if skipped_existing else "")
        + (f" (skipped {skipped_no_image} with no matching image)" if skipped_no_image else "")
    )
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=list(SPLITS.keys()) + ["all"],
        default="val",
        help="which split to convert (default: val)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing label files",
    )
    args = parser.parse_args()

    splits = list(SPLITS.keys()) if args.split == "all" else [args.split]
    for s in splits:
        convert_split(s, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
