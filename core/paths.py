"""Central project paths (repo root = parent of core/)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Shared libraries
CORE_DIR = PROJECT_ROOT / "core"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Data (large image trees stay at dataset/ and kitti/ — see docs/FOLDER_LAYOUT.md)
DATA_DEMO = PROJECT_ROOT / "data" / "demo"
DATASET_VISDRONE_DIR = PROJECT_ROOT / "dataset"
DATASET_VISDRONE_YAML = DATASET_VISDRONE_DIR / "VisDrone.yaml"
DATASET_KITTI_DIR = PROJECT_ROOT / "kitti"
DATASET_KITTI_YAML = DATASET_KITTI_DIR / "kitti.yaml"

# Model weights (gitignored *.pt)
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
YOLO_VISDRONE = RUNS_DIR / "detect" / "train9" / "weights" / "best.pt"
YOLO_VISDRONE_V8 = RUNS_DIR / "detect" / "train_v8_visdrone" / "weights" / "best.pt"
YOLO_KITTI = RUNS_DIR / "detect" / "train2" / "weights" / "best.pt"
YOLO_STOCK = PROJECT_ROOT / "yolo11n.pt"
SAM3_WEIGHTS = PROJECT_ROOT / "sam3.pt"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_EXPERIMENTS = OUTPUTS_DIR / "experiments"
OUTPUT_VALIDATION = OUTPUTS_DIR / "validation"
