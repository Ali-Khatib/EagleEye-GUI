from __future__ import annotations

import argparse
import csv
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.yolo_weights import resolve_yolo_checkpoint  # noqa: E402

OUTPUT_ROOT = ROOT / "benchmark_working" / "yolo_validation"

# Official dataset-level validation configs. These are the only metrics that
# should be treated as research-paper detection metrics.
DATASETS = {
    "visdrone": {
        "model": ROOT / resolve_yolo_checkpoint("visdrone", warn=False),
        "data": ROOT / "dataset" / "VisDrone.yaml",
    },
    "kitti": {
        "model": ROOT / resolve_yolo_checkpoint("kitti", warn=False),
        "data": ROOT / "kitti" / "kitti.yaml",
    },
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_split_dir(data_yaml: Path, cfg: dict[str, Any], split: str) -> Path:
    """Resolve train/val image directories using Ultralytics-style data YAML."""
    base = Path(str(cfg.get("path", "")))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    split_value = cfg.get(split)
    if split_value is None:
        raise ValueError(f"Missing '{split}' entry in {data_yaml}")
    split_path = Path(str(split_value))
    if not split_path.is_absolute():
        split_path = base / split_path
    return split_path.resolve()


def infer_label_dir(image_dir: Path) -> Path:
    """Ultralytics expects labels beside images with /images/ replaced by /labels/."""
    parts = list(image_dir.parts)
    if "images" not in parts:
        raise ValueError(
            f"Could not infer label dir from image dir because it has no 'images' segment: {image_dir}"
        )
    idx = parts.index("images")
    parts[idx] = "labels"
    return Path(*parts)


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("data.yaml 'names' must be a dict or list")


def verify_dataset_structure(data_yaml: Path, model: YOLO) -> dict[str, Any]:
    """Verify image/label pairing and YAML class names before validation."""
    cfg = read_yaml(data_yaml)
    names = normalize_names(cfg.get("names"))
    model_names = {int(k): str(v) for k, v in model.names.items()}
    if names != model_names:
        raise ValueError(
            "data.yaml class names do not match the trained model.\n"
            f"YAML names:  {names}\n"
            f"Model names: {model_names}"
        )

    val_images = resolve_split_dir(data_yaml, cfg, "val")
    train_images = resolve_split_dir(data_yaml, cfg, "train")
    val_labels = infer_label_dir(val_images)
    train_labels = infer_label_dir(train_images)

    problems: list[str] = []
    for label, path in (
        ("train images", train_images),
        ("val images", val_images),
        ("train labels", train_labels),
        ("val labels", val_labels),
    ):
        if not path.is_dir():
            problems.append(f"Missing {label} directory: {path}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    val_image_files = sorted(
        p for p in val_images.glob("*") if p.suffix.lower() in image_exts
    ) if val_images.is_dir() else []
    val_label_files = sorted(val_labels.glob("*.txt")) if val_labels.is_dir() else []

    missing_labels = []
    if val_images.is_dir() and val_labels.is_dir():
        label_stems = {p.stem for p in val_label_files}
        for img in val_image_files:
            if img.stem not in label_stems:
                missing_labels.append(img.name)
                if len(missing_labels) >= 20:
                    break

    if not val_image_files:
        problems.append(f"No validation images found in {val_images}")
    if not val_label_files:
        problems.append(f"No validation labels found in {val_labels}")
    if missing_labels:
        problems.append(
            f"{len(missing_labels)}+ validation images have no matching label. "
            f"Examples: {missing_labels}"
        )

    if problems:
        raise ValueError("Dataset verification failed:\n- " + "\n- ".join(problems))

    return {
        "data_yaml": str(data_yaml),
        "train_images": str(train_images),
        "train_labels": str(train_labels),
        "val_images": str(val_images),
        "val_labels": str(val_labels),
        "val_image_count": len(val_image_files),
        "val_label_count": len(val_label_files),
        "class_count": len(names),
        "names": names,
    }


def hardware_info() -> dict[str, str]:
    cuda_available = torch.cuda.is_available()
    device = "cuda:0" if cuda_available else "cpu"
    gpu = torch.cuda.get_device_name(0) if cuda_available else "CPU only"
    return {
        "gpu": gpu,
        "cuda_available": str(cuda_available),
        "cuda_version": torch.version.cuda or "N/A",
        "pytorch_version": torch.__version__,
        "device_used": device,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def f1_from_pr(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def save_metrics(dataset: str, metrics: dict[str, Any], verification: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "results.csv"
    txt_path = OUTPUT_ROOT / "results.txt"

    row = {"dataset": dataset, **metrics}
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    with txt_path.open("a", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"Dataset: {dataset}\n")
        f.write("Official YOLO validation metrics (research-paper source)\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nDataset verification:\n")
        for k, v in verification.items():
            f.write(f"{k}: {v}\n")
        f.write("\nMetric meanings:\n")
        f.write("Precision: fraction of predicted boxes that are correct.\n")
        f.write("Recall: fraction of ground-truth objects detected.\n")
        f.write("mAP50: mean average precision at IoU threshold 0.50.\n")
        f.write("mAP50-95: mean AP averaged from IoU 0.50 to 0.95; stricter.\n")
        f.write("F1: harmonic mean of precision and recall.\n")
        f.write(
            "\nSingle-image dashboard experiments are for visual comparison, FPS, "
            "runtime, and qualitative SAHI/SAM behavior only. They do not replace "
            "official validation on the full validation set.\n\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Official Ultralytics validation for research-paper metrics."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="visdrone")
    parser.add_argument("--model", type=Path, default=None, help="Override model path.")
    parser.add_argument("--data", type=Path, default=None, help="Override data.yaml path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    args = parser.parse_args()

    default = DATASETS[args.dataset]
    model_path = (args.model or default["model"]).resolve()
    data_yaml = (args.data or default["data"]).resolve()
    out_dir = OUTPUT_ROOT / args.dataset

    print("Official YOLO validation metrics are the real research-paper metrics.")
    print("Custom single-image SAHI/SAM experiments are visual/FPS/runtime demos only.")
    print()

    hw = hardware_info()
    for k, v in hw.items():
        print(f"{k}: {v}")
    if hw["device_used"] == "cpu":
        print("[WARNING] CUDA is unavailable. Validation will run on CPU and be slower.")
    print()

    if not model_path.is_file():
        print(f"[ERROR] Model not found: {model_path}")
        return 1
    if not data_yaml.is_file():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        return 1

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    try:
        verification = verify_dataset_structure(data_yaml, model)
    except Exception as ex:
        print(f"[ERROR] {ex}")
        return 1

    print("Dataset verification passed:")
    print(f"  val images: {verification['val_image_count']} -> {verification['val_images']}")
    print(f"  val labels: {verification['val_label_count']} -> {verification['val_labels']}")
    print(f"  classes: {verification['class_count']} -> {verification['names']}")
    print()

    # Equivalent to:
    #   yolo detect val model=<model> data=<data.yaml> project=validation_results name=<dataset>
    # This saves confusion matrix, PR curves, predictions, plots, and CSV files
    # inside validation_results/<dataset>/.
    results = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(OUTPUT_ROOT),
        name=args.dataset,
        exist_ok=True,
        plots=True,
        save_json=True,
        save_txt=True,
        save_conf=True,
    )

    precision = float(results.box.mp)
    recall = float(results.box.mr)
    map50 = float(results.box.map50)
    map5095 = float(results.box.map)
    f1 = f1_from_pr(precision, recall)

    metrics = {
        "model": str(model_path),
        "data_yaml": str(data_yaml),
        "device_used": hw["device_used"],
        "gpu": hw["gpu"],
        "cuda_available": hw["cuda_available"],
        "cuda_version": hw["cuda_version"],
        "pytorch_version": hw["pytorch_version"],
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "map50": round(map50, 6),
        "map50_95": round(map5095, 6),
        "validation_output_dir": str(out_dir),
    }

    print("\nOfficial validation metrics:")
    print(f"Precision : {metrics['precision']}")
    print(f"Recall    : {metrics['recall']}")
    print(f"F1        : {metrics['f1']}")
    print(f"mAP50     : {metrics['map50']}")
    print(f"mAP50-95  : {metrics['map50_95']}")
    print(f"\nValidation plots/predictions saved to: {out_dir}")

    save_metrics(args.dataset, metrics, verification)
    print(f"Saved summary CSV: {OUTPUT_ROOT / 'results.csv'}")
    print(f"Saved summary TXT: {OUTPUT_ROOT / 'results.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
