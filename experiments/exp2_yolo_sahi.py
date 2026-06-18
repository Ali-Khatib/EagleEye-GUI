"""Pipeline 2: YOLO + SAHI."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from experiments.demo_runner import run_demo

if __name__ == "__main__":
    run_demo(
        "yolo_sahi",
        mode_name="YOLO + SAHI",
        pipeline_folder="yolo sahi",
        class_agnostic=False,
        needs_sam3=False,
    )
