"""Pipeline 3: SAM 3 only (text prompts)."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from experiments.demo_runner import run_demo

if __name__ == "__main__":
    run_demo(
        "sam3_only",
        mode_name="SAM 3 only",
        pipeline_folder="sam3 only",
        class_agnostic=True,
        needs_sam3=True,
    )
