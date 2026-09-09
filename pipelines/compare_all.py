#!/usr/bin/env python3
"""
Fair comparison of five pipelines (YOLO + SAHI + SAM 3).

  yolo_only        — YOLO baseline
  yolo_sahi        — YOLO + SAHI
  sam3_only        — SAM 3 text prompts
  sam3_yolo        — YOLO detect + SAM 3 mask refine
  yolo_sahi_sam3   — YOLO + SAHI + SAM 3 (main pipeline)

Usage:
  python pipelines/compare_all.py \\
    --source path/to/images \\
    --weights runs/detect/train9/weights/best.pt \\
    --ground-truth path/to/labels \\
    --classes person pedestrian car van truck bus bicycle motorcycle
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.compare_metrics import (  # noqa: E402
    aggregate_mode,
    append_benchmark_checkpoint,
    clear_benchmark_checkpoint,
    compute_map50_95_pooled,
    compute_map50_pooled,
    evaluate_image,
    load_benchmark_checkpoint,
    load_mode_aggregate_from_metrics,
    mode_benchmark_complete,
    recommend_mode,
    save_mode_outputs,
    write_comparison_csv,
    write_supervisor_report_csv,
)
from core.pipeline_config import SAHI_CONF, SAHI_YOLO_IMGSZ  # noqa: E402
from core.sahi_model_cache import preload_sahi_detection_model  # noqa: E402
from core.compare_viz import draw_detections  # noqa: E402
from core.detection_ops import detections_to_yolo_txt  # noqa: E402
from core.eval_config import EVAL_CONF, EVAL_IOU  # noqa: E402
from core.sam3_support import infer_sam3_dataset, resolve_sam3_weights  # noqa: E402
from core.yolo_weights import resolve_yolo_checkpoint  # noqa: E402
from pipelines._common import DEFAULT_TEXT_CLASSES, PipelineResult  # noqa: E402
from pipelines import (  # noqa: E402
    sam3_only,
    sam3_yolo,
    yolo_only,
    yolo_sahi,
    yolo_sahi_sam3,
)

ALL_MODES = ("yolo_only", "yolo_sahi", "sam3_only", "sam3_yolo", "yolo_sahi_sam3")
SAM3_MODES = frozenset({"sam3_only", "sam3_yolo", "yolo_sahi_sam3"})


def write_progress_log(
    path: Optional[Path],
    *,
    mode: str,
    done: int,
    total: int,
    image_name: str,
    started_at: float,
) -> None:
    if path is None:
        return
    elapsed = max(time.time() - started_at, 0.001)
    rate = done / elapsed
    remaining = max(total - done, 0)
    eta_s = int(remaining / rate) if rate > 0 else 0
    eta_h, rem = divmod(eta_s, 3600)
    eta_m, eta_s = divmod(rem, 60)
    pct = 100.0 * done / total if total else 0.0
    bar_w = 30
    filled = int(bar_w * done / total) if total else 0
    bar = "#" * filled + "-" * (bar_w - filled)
    line = (
        f"{mode} |{bar}| {done}/{total} ({pct:5.1f}%) "
        f"{image_name[:28]} ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line + "\n", encoding="utf-8")


def discover_pairs(source: Path, gt_dir: Path) -> List[Tuple[Path, Path]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
    pairs: List[Tuple[Path, Path]] = []
    for img in images:
        lbl = gt_dir / f"{img.stem}.txt"
        if lbl.is_file():
            pairs.append((img, lbl))
        else:
            print(f"[WARN] No label for {img.name}, skipping.")
    return pairs


def run_mode(
    mode: str,
    pairs: List[Tuple[Path, Path]],
    yolo_model: YOLO,
    yolo_weights: str,
    text_classes: List[str],
    device: str,
    sam3_path: Path | None,
    out_dir: Path,
    class_names: dict,
    max_vis: int = 5,
    skip_images: bool = False,
    save_labels: bool = False,
    progress: tqdm | None = None,
    progress_log: Optional[Path] = None,
    progress_started_at: float = 0.0,
    global_done: int = 0,
    global_total: int = 0,
    resume: bool = False,
):
    per_image: List[object] = []
    map50_vals: List[float] = []
    map50_95_vals: List[float] = []
    notes = ""
    class_agnostic = False
    done_names: set[str] = set()
    if resume:
        per_image, map50_vals, map50_95_vals, notes, done_names = load_benchmark_checkpoint(out_dir)
        if done_names:
            print(f"[RESUME] {mode}: {len(done_names)} images loaded from checkpoint")

    if progress is not None and done_names:
        progress.update(len(done_names))

    if not skip_images:
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    processed_count = len(done_names)
    for img_path, lbl_path in pairs:
        if img_path.name in done_names:
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            if progress is not None:
                progress.update(1)
            continue
        h, w = image_bgr.shape[:2]

        if mode == "yolo_only":
            out: PipelineResult = yolo_only.run(image_bgr, yolo_model, device)
        elif mode == "yolo_sahi":
            out = yolo_sahi.run(image_bgr, yolo_weights, device)
        elif mode == "sam3_only":
            out = sam3_only.run(
                image_bgr, text_classes, device,
                sam3_weights=sam3_path,
                class_names=class_names,
            )
        elif mode == "sam3_yolo":
            out = sam3_yolo.run(
                image_bgr, yolo_model, device,
                sam3_weights=sam3_path,
                class_names=class_names,
            )
        elif mode == "yolo_sahi_sam3":
            out = yolo_sahi_sam3.run(
                image_bgr, yolo_weights, device,
                yolo_model=yolo_model,
                sam3_weights=sam3_path,
                class_names=class_names,
            )
        else:
            raise ValueError(mode)

        notes = out.notes
        im = evaluate_image(
            img_path, lbl_path, out.detections, out.runtime_seconds,
            class_agnostic, masks=out.masks, pipeline_stats=out.stats,
            image_hw=(h, w),
        )
        per_image.append(im)

        from core.experiment_utils import load_ground_truth_yolo_labels_checked
        gt_r = load_ground_truth_yolo_labels_checked(str(lbl_path), w, h, image_path=str(img_path))
        map50_val: float | None = None
        map50_95_val: float | None = None
        if gt_r["reliable"]:
            gt_boxes = gt_r["boxes"]
            map50_val = compute_map50_pooled(out.detections, gt_boxes, class_agnostic=class_agnostic)
            map50_95_val = compute_map50_95_pooled(out.detections, gt_boxes, class_agnostic=class_agnostic)
            map50_vals.append(map50_val)
            map50_95_vals.append(map50_95_val)

        append_benchmark_checkpoint(out_dir, im, map50_val, map50_95_val, notes)
        processed_count += 1

        if save_labels or not skip_images:
            (out_dir / "labels").mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / f"{img_path.stem}.txt").write_text(
                detections_to_yolo_txt(out.detections, w, h), encoding="utf-8",
            )
        if not skip_images:
            for mi, mask in enumerate(out.masks):
                if mask is None:
                    continue
                cv2.imwrite(str(out_dir / "masks" / f"{img_path.stem}_{mi}.png"), mask * 255)
            vis = draw_detections(image_bgr, out.detections, class_names, out.masks or None)
            cv2.imwrite(str(out_dir / "images" / img_path.name), vis)
            if processed_count <= max_vis:
                cv2.imwrite(str(out_dir / "vis" / img_path.name), vis)

        if progress is not None:
            progress.set_postfix(mode=mode, image=img_path.name[:24], refresh=False)
            progress.update(1)
        write_progress_log(
            progress_log,
            mode=mode,
            done=global_done + processed_count,
            total=global_total,
            image_name=img_path.name,
            started_at=progress_started_at,
        )

    if progress is not None:
        progress.set_postfix(mode=mode, stage="aggregate", refresh=False)
    agg = aggregate_mode(mode, per_image, notes)
    agg.map50 = sum(map50_vals) / len(map50_vals) if map50_vals else 0.0
    agg.map50_95 = sum(map50_95_vals) / len(map50_95_vals) if map50_95_vals else 0.0
    save_mode_outputs(out_dir, agg)
    clear_benchmark_checkpoint(out_dir)
    return agg


def export_labels_mode(
    mode: str,
    pairs: List[Tuple[Path, Path]],
    yolo_model: YOLO,
    yolo_weights: str,
    text_classes: List[str],
    device: str,
    sam3_path: Path | None,
    out_dir: Path,
    class_names: dict,
    progress: tqdm | None = None,
    resume: bool = False,
) -> int:
    """Run inference once and save YOLO prediction txt only (no metrics/images)."""
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0

    for img_path, _lbl_path in pairs:
        pred_path = labels_dir / f"{img_path.stem}.txt"
        if resume and pred_path.is_file():
            skipped += 1
            if progress is not None:
                progress.update(1)
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            if progress is not None:
                progress.update(1)
            continue
        h, w = image_bgr.shape[:2]

        if mode == "yolo_only":
            out: PipelineResult = yolo_only.run(image_bgr, yolo_model, device)
        elif mode == "yolo_sahi":
            out = yolo_sahi.run(image_bgr, yolo_weights, device)
        elif mode == "sam3_only":
            out = sam3_only.run(
                image_bgr, text_classes, device,
                sam3_weights=sam3_path,
                class_names=class_names,
            )
        elif mode == "sam3_yolo":
            out = sam3_yolo.run(
                image_bgr, yolo_model, device,
                sam3_weights=sam3_path,
                class_names=class_names,
            )
        elif mode == "yolo_sahi_sam3":
            out = yolo_sahi_sam3.run(
                image_bgr, yolo_weights, device,
                yolo_model=yolo_model,
                sam3_weights=sam3_path,
                class_names=class_names,
            )
        else:
            raise ValueError(mode)

        pred_path.write_text(
            detections_to_yolo_txt(out.detections, w, h), encoding="utf-8",
        )
        saved += 1
        if progress is not None:
            progress.set_postfix(mode=mode, image=img_path.name[:24], refresh=False)
            progress.update(1)

    print(f"[LABELS] {mode}: saved {saved}, skipped {skipped} (already exported)")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare YOLO / SAHI / SAM 3 pipelines.")
    parser.add_argument("--source", required=True, help="Image directory")
    parser.add_argument("--weights", default="", help="YOLO .pt weights")
    parser.add_argument("--ground-truth", required=True, help="YOLO label directory")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_TEXT_CLASSES,
        help="SAM 3 text prompts",
    )
    parser.add_argument("--output", default="outputs", help="Output root (mode subfolders created)")
    parser.add_argument("--modes", default=",".join(ALL_MODES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sam3-weights", default="", help="Path to sam3.pt")
    parser.add_argument("--device", default="", help="cuda or cpu")
    parser.add_argument("--skip-images", action="store_true", help="Skip saving images/masks (faster)")
    parser.add_argument(
        "--save-labels",
        action="store_true",
        help="Save YOLO prediction txt per image even when --skip-images is set",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only export prediction txt files (no metrics, no vis). Use with --save-labels.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip modes that already have metrics.json for all images",
    )
    parser.add_argument("--dataset-name", default="", help="Label for supervisor report CSV")
    parser.add_argument("--data-yaml", default="", help="Optional: Ultralytics val mAP for yolo_only")
    parser.add_argument(
        "--progress-log",
        default="",
        help="Write one-line tqdm-style progress to this file (for watch scripts)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from per-image checkpoint if benchmark was interrupted",
    )
    args = parser.parse_args()

    source = Path(args.source)
    gt_dir = Path(args.ground_truth)
    out_root = Path(args.output)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    pairs = discover_pairs(source, gt_dir)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        print("[ERROR] No image/label pairs.")
        sys.exit(1)

    yolo_weights = args.weights or resolve_yolo_checkpoint("visdrone")
    wp = Path(yolo_weights)
    if not wp.is_file():
        wp = ROOT / yolo_weights
    if not wp.is_file():
        print(f"[ERROR] YOLO weights not found: {yolo_weights}")
        sys.exit(1)
    yolo_weights = str(wp)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[ERROR] --device cuda requested but torch.cuda.is_available() is False.")
        sys.exit(1)
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    sam3_path: Path | None = None
    sam3_dataset = infer_sam3_dataset(source, yolo_weights, args.dataset_name)
    if SAM3_MODES.intersection(modes):
        sam3_path = resolve_sam3_weights(args.sam3_weights or None, dataset=sam3_dataset)
        if sam3_dataset and (ROOT / "runs" / "sam3_finetune" / sam3_dataset / "sam3.pt").is_file():
            print(f"[INFO] SAM3 fine-tuned ({sam3_dataset}): {sam3_path}")
        else:
            print(f"[INFO] SAM3: {sam3_path}")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] YOLO: {yolo_weights}")
    print(f"[INFO] conf={EVAL_CONF} iou={EVAL_IOU}")
    print(f"[INFO] SAM3 classes: {args.classes}")
    print(f"[INFO] {len(pairs)} images, modes: {modes}")

    yolo_model = YOLO(yolo_weights)
    class_names = yolo_model.names

    if {"yolo_sahi", "yolo_sahi_sam3"}.intersection(modes):
        preload_sahi_detection_model(
            yolo_weights,
            device,
            confidence_threshold=SAHI_CONF,
            image_size=SAHI_YOLO_IMGSZ,
        )
        from core.pipeline_config import SAHI_OVERLAP, SAHI_SLICE

        print(
            f"[INFO] SAHI cached (slice={SAHI_SLICE}px overlap={SAHI_OVERLAP:.0%} "
            f"imgsz={SAHI_YOLO_IMGSZ})"
        )

    aggregates: Dict[str, object] = {}
    pending_modes: List[str] = []
    for mode in modes:
        if mode not in ALL_MODES:
            print(f"[ERROR] Unknown mode {mode}")
            sys.exit(1)
        mode_dir = out_root / mode
        if args.labels_only:
            pending_modes.append(mode)
            continue
        if args.skip_existing and mode_benchmark_complete(mode_dir, len(pairs)):
            loaded = load_mode_aggregate_from_metrics(mode_dir)
            if loaded:
                aggregates[mode] = loaded
                print(
                    f"[SKIP] {mode} — already done "
                    f"({loaded.images_evaluated} images, F1={loaded.f1:.3f})"
                )
                continue
        pending_modes.append(mode)

    total_steps = len(pending_modes) * len(pairs)
    progress_log = Path(args.progress_log) if args.progress_log else None
    progress_started_at = time.time()
    images_done = 0
    progress = tqdm(
        total=total_steps,
        desc="Benchmark",
        unit="img",
        dynamic_ncols=True,
        disable=total_steps == 0,
    )

    for mode in pending_modes:
        print(f"\n{'='*60}\n{mode}\n{'='*60}")
        mode_dir = out_root / mode
        try:
            if args.labels_only:
                export_labels_mode(
                    mode, pairs, yolo_model, yolo_weights, args.classes,
                    device, sam3_path, mode_dir, class_names,
                    progress=progress,
                    resume=args.resume,
                )
                images_done += len(pairs)
                continue
            agg = run_mode(
                mode, pairs, yolo_model, yolo_weights, args.classes,
                device, sam3_path, mode_dir, class_names,
                skip_images=args.skip_images,
                save_labels=args.save_labels,
                progress=progress,
                progress_log=progress_log,
                progress_started_at=progress_started_at,
                global_done=images_done,
                global_total=total_steps,
                resume=args.resume,
            )
            images_done += len(pairs)
            aggregates[mode] = agg
            print(
                f"  F1={agg.f1:.3f} P={agg.precision:.3f} R={agg.recall:.3f} "
                f"mAP50~={agg.map50:.3f} small_R={agg.small_object_recall:.3f} "
                f"mask_q={agg.mask_quality:.3f} FPS={agg.fps:.2f}"
            )
        except FileNotFoundError as exc:
            progress.close()
            print(f"[ERROR] {exc}")
            sys.exit(1)

    progress.close()

    if args.labels_only:
        print(f"\n[DONE] Prediction labels under {out_root}/<mode>/labels/")
        return

    if args.skip_existing:
        for mode in ALL_MODES:
            if mode in aggregates:
                continue
            loaded = load_mode_aggregate_from_metrics(out_root / mode)
            if loaded:
                aggregates[mode] = loaded

    if args.data_yaml and "yolo_only" in aggregates:
        try:
            from ultralytics import YOLO as _Y
            m = _Y(yolo_weights)
            met = m.val(data=str(args.data_yaml), split="val", device=0 if device == "cuda" else "cpu", verbose=False)
            aggregates["yolo_only"].map50 = float(met.box.map50)
            save_mode_outputs(out_root / "yolo_only", aggregates["yolo_only"], aggregates["yolo_only"].map50)
            print(f"[INFO] Official YOLO mAP50: {aggregates['yolo_only'].map50:.4f}")
        except Exception as exc:
            print(f"[WARN] Ultralytics val: {exc}")

    csv_path = out_root / "comparison_results.csv"
    write_comparison_csv(csv_path, aggregates)
    if args.dataset_name:
        write_supervisor_report_csv(
            out_root / "supervisor_metrics.csv", args.dataset_name, aggregates,
        )
    rec = recommend_mode(aggregates)
    (out_root / "recommendation.txt").write_text(rec["recommendation"], encoding="utf-8")
    (out_root / "best_mode.txt").write_text(
        f"Suggested best overall: {rec.get('best_overall', 'yolo_sahi_sam3')}\n\n{rec['recommendation']}",
        encoding="utf-8",
    )
    print(f"\n[DONE] {csv_path}")
    print(rec["recommendation"])


if __name__ == "__main__":
    main()
