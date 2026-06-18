"""Pipeline 4: SAM 3 + YOLO."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from experiments.demo_runner import run_demo

if __name__ == "__main__":
    run_demo(
        "sam3_yolo",
        mode_name="SAM 3 + YOLO",
        pipeline_folder="sam3 yolo",
        class_agnostic=False,
        needs_sam3=True,
    )
