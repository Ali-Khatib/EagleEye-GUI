"""Pipeline 5: YOLO + SAHI + SAM 3 (main pipeline)."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from experiments.demo_runner import run_demo

if __name__ == "__main__":
    run_demo(
        "yolo_sahi_sam3",
        mode_name="YOLO + SAHI + SAM 3",
        pipeline_folder="yolo sahi sam3",
        class_agnostic=False,
        needs_sam3=True,
    )
