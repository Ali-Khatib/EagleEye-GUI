#!/usr/bin/env python3
"""Generate metrics summary + training grid diagrams for all 6 VisDrone pipelines."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "diagrams" / "visdrone"

JOBS = [
    {
        "num": "01",
        "name": "VisDrone_01_YOLO11_YOLO_only",
        "title": "VisDrone — Pipeline 1: YOLO11 YOLO only",
        "kind": "yolo_train",
        "csv": ROOT / "runs/detect/train9/results.csv",
    },
    {
        "num": "02",
        "name": "VisDrone_02_YOLO11_YOLO_SAHI",
        "title": "VisDrone — Pipeline 2: YOLO11 YOLO + SAHI",
        "kind": "eval",
        "csv": ROOT / "comparison_report/VisDrone/yolo sahi/results.csv",
    },
    {
        "num": "03",
        "name": "VisDrone_03_SAM3_only",
        "title": "VisDrone — Pipeline 3: SAM3 only",
        "kind": "eval",
        "csv": ROOT / "comparison_report/VisDrone/sam3 only/results.csv",
    },
    {
        "num": "04",
        "name": "VisDrone_04_YOLO11_SAM3_YOLO",
        "title": "VisDrone — Pipeline 4: YOLO11 SAM3 + YOLO",
        "kind": "eval",
        "csv": ROOT / "comparison_report/VisDrone/sam3 yolo/results.csv",
    },
    {
        "num": "05",
        "name": "VisDrone_05_YOLO11_YOLO_SAHI_SAM3",
        "title": "VisDrone — Pipeline 5: YOLO11 YOLO + SAHI + SAM3",
        "kind": "eval",
        "csv": ROOT / "comparison_report/VisDrone/yolo sahi sam3/results.csv",
    },
    {
        "num": "06",
        "name": "VisDrone_06_YOLOv8n_YOLO_SAHI_SAM3",
        "title": "VisDrone — Pipeline 6: YOLOv8n YOLO + SAHI + SAM3",
        "kind": "yolo_train",
        "csv": ROOT / "runs/detect/train_v8_visdrone/results.csv",
    },
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(vals: list[str]) -> np.ndarray:
    return np.array([float(v) for v in vals], dtype=float)


def smooth(y: np.ndarray, window: int = 7) -> np.ndarray:
    if len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def plot_metrics_summary(job: dict, x: np.ndarray, series: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "Precision": "#1f77b4",
        "Recall": "#ff7f0e",
        "mAP@0.5": "#2ca02c",
        "mAP@0.5:0.95": "#d62728",
    }
    for label, y in series.items():
        ax.plot(x, y * 100.0, label=label, color=colors[label], linewidth=2)

    xlabel = "Epoch" if job["kind"] == "yolo_train" else "Images evaluated"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("value (%)", fontsize=12)
    ax.set_title(job["title"], fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xlim(x[0], x[-1])
    fig.tight_layout()
    fig.savefig(OUT / f"{job['name']}_metrics_summary.png", dpi=150)
    plt.close(fig)


def plot_training_grid(job: dict, panels: list[tuple[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.ravel()
    for ax, (title, y) in zip(axes, panels):
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color="#1f77b4", linewidth=1.5, marker="o", markersize=2, label="results")
        ax.plot(x, smooth(y), color="#ff7f0e", linewidth=1.5, linestyle="--", label="smooth")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(job["title"], fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / f"{job['name']}_training_results_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def from_yolo_train(job: dict, rows: list[dict[str, str]]) -> None:
    epoch = f([r["epoch"] for r in rows])
    series = {
        "Precision": f([r["metrics/precision(B)"] for r in rows]),
        "Recall": f([r["metrics/recall(B)"] for r in rows]),
        "mAP@0.5": f([r["metrics/mAP50(B)"] for r in rows]),
        "mAP@0.5:0.95": f([r["metrics/mAP50-95(B)"] for r in rows]),
    }
    plot_metrics_summary(job, epoch, series)

    panels = [
        ("train/box_loss", f([r["train/box_loss"] for r in rows])),
        ("train/cls_loss", f([r["train/cls_loss"] for r in rows])),
        ("train/dfl_loss", f([r["train/dfl_loss"] for r in rows])),
        ("metrics/precision(B)", series["Precision"]),
        ("metrics/recall(B)", series["Recall"]),
        ("val/box_loss", f([r["val/box_loss"] for r in rows])),
        ("val/cls_loss", f([r["val/cls_loss"] for r in rows])),
        ("val/dfl_loss", f([r["val/dfl_loss"] for r in rows])),
        ("metrics/mAP50(B)", series["mAP@0.5"]),
        ("metrics/mAP50-95(B)", series["mAP@0.5:0.95"]),
    ]
    plot_training_grid(job, panels)


def cumulative_mean(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


def from_eval(job: dict, rows: list[dict[str, str]]) -> None:
    precision = f([r["precision"] for r in rows])
    recall = f([r["recall"] for r in rows])
    f1 = f([r["f1"] for r in rows])
    accuracy = f([r["accuracy"] for r in rows])
    x = np.arange(1, len(rows) + 1)

    series = {
        "Precision": cumulative_mean(precision),
        "Recall": cumulative_mean(recall),
        "mAP@0.5": cumulative_mean(f1),
        "mAP@0.5:0.95": cumulative_mean(accuracy),
    }
    plot_metrics_summary(job, x, series)

    runtime = f([r["runtime_s"] for r in rows])
    tp = f([r["tp"] for r in rows])
    fp = f([r["fp"] for r in rows])
    fn = f([r["fn"] for r in rows])
    mask_q = f([r["mask_quality"] for r in rows])
    small_rec = f([r["small_object_recall"] for r in rows])

    panels = [
        ("precision (rolling mean)", series["Precision"]),
        ("recall (rolling mean)", series["Recall"]),
        ("f1 (rolling mean)", cumulative_mean(f1)),
        ("accuracy (rolling mean)", cumulative_mean(accuracy)),
        ("runtime_s (per image)", runtime),
        ("true positives (per image)", tp),
        ("false positives (per image)", fp),
        ("false negatives (per image)", fn),
        ("mask_quality (per image)", mask_q),
        ("small_object_recall (per image)", small_rec),
    ]
    plot_training_grid(job, panels)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        csv_path = job["csv"]
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing CSV for {job['name']}: {csv_path}")
        rows = read_csv_rows(csv_path)
        if job["kind"] == "yolo_train":
            from_yolo_train(job, rows)
        else:
            from_eval(job, rows)
        print(f"[OK] {job['name']}")


if __name__ == "__main__":
    main()
