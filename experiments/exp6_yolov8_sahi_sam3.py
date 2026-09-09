"""Pipeline 6: YOLOv8 + SAHI + SAM 3 (VisDrone paper comparison backbone)."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.paths import YOLO_VISDRONE_V8
from experiments.demo_runner import run_demo

if __name__ == "__main__":
    run_demo(
        "yolo_sahi_sam3",
        mode_name="YOLOv8 + SAHI + SAM 3",
        pipeline_folder="yolov8 sahi sam3",
        class_agnostic=False,
        needs_sam3=True,
        yolo_weights_override=str(YOLO_VISDRONE_V8),
    )
