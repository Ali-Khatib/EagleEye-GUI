#!/usr/bin/env python3
"""Run fair comparison of all five SAM 3 pipelines."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.compare_all import main  # noqa: E402

if __name__ == "__main__":
    main()
