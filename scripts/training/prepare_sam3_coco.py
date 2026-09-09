#!/usr/bin/env python3
"""
Convert YOLO images + labels to COCO format for Meta SAM3 fine-tuning.

Output layout (Roboflow-compatible):
  data/sam3_coco/{dataset}/
    train/images/ + train/_annotations.coco.json
    val/images/   + val/_annotations.coco.json

Segmentation uses rectangle polygons from YOLO boxes (pseudo-masks).
Meta SAM3 training expects 1008px inputs; images are copied as-is (resize in train config).

Usage:
  python scripts/training/prepare_sam3_coco.py --dataset visdrone
  python scripts/training/prepare_sam3_coco.py --dataset kitti --limit 200
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_SPLITS = {
    "visdrone": {
        "train_images": ROOT / "dataset" / "VisDrone2019-DET-train" / "VisDrone2019-DET-train" / "images",
        "train_labels": ROOT / "dataset" / "VisDrone2019-DET-train" / "VisDrone2019-DET-train" / "labels",
        "val_images": ROOT / "dataset" / "VisDrone2019-DET-val" / "VisDrone2019-DET-val" / "images",
        "val_labels": ROOT / "dataset" / "VisDrone2019-DET-val" / "VisDrone2019-DET-val" / "labels",
        "names": {
            0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
            5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
        },
    },
    "kitti": {
        "train_images": ROOT / "kitti" / "images" / "train",
        "train_labels": ROOT / "kitti" / "labels" / "train",
        "val_images": ROOT / "kitti" / "images" / "val",
        "val_labels": ROOT / "kitti" / "labels" / "val",
        "names": {
            0: "car", 1: "van", 2: "truck", 3: "pedestrian", 4: "person_sitting",
            5: "cyclist", 6: "tram", 7: "misc",
        },
    },
}


def _yolo_to_coco_split(
    images_dir: Path,
    labels_dir: Path,
    names: dict[int, str],
    out_dir: Path,
    limit: int,
) -> dict:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if limit > 0:
        images = images[:limit]

    out_img = out_dir / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid + 1, "name": name} for cid, name in names.items()],
    }
    ann_id = 1
    for img_id, src in enumerate(images, start=1):
        lbl = labels_dir / f"{src.stem}.txt"
        if not lbl.is_file():
            continue
        dst = out_img / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

        import cv2
        im = cv2.imread(str(src))
        if im is None:
            continue
        h, w = im.shape[:2]
        coco["images"].append({
            "id": img_id,
            "file_name": src.name,
            "width": w,
            "height": h,
        })

        for line in lbl.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            bw_px = bw * w
            bh_px = bh * h
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw_px), min(h, y + bh_px)
            seg = [x1, y1, x2, y1, x2, y2, x1, y2]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "segmentation": [seg],
                "iscrowd": 0,
            })
            ann_id += 1

    ann_path = out_dir / "_annotations.coco.json"
    ann_path.write_text(json.dumps(coco), encoding="utf-8")
    return {"images": len(coco["images"]), "annotations": len(coco["annotations"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare COCO data for SAM3 fine-tune")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPLITS), required=True)
    parser.add_argument("--limit", type=int, default=0, help="Max images per split (0=all)")
    parser.add_argument(
        "--output",
        default="",
        help="Output root (default: data/sam3_coco/{dataset})",
    )
    args = parser.parse_args()

    cfg = DATASET_SPLITS[args.dataset]
    out_root = Path(args.output) if args.output else ROOT / "data" / "sam3_coco" / args.dataset

    for split, img_key, lbl_key in (
        ("train", "train_images", "train_labels"),
        ("val", "val_images", "val_labels"),
    ):
        stats = _yolo_to_coco_split(
            cfg[img_key], cfg[lbl_key], cfg["names"], out_root / split, args.limit,
        )
        print(f"[OK] {args.dataset}/{split}: {stats['images']} images, {stats['annotations']} boxes")
    print(f"[DONE] COCO data at {out_root}")


if __name__ == "__main__":
    main()
