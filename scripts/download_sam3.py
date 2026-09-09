#!/usr/bin/env python3
"""
Download sam3.pt from Hugging Face (facebook/sam3) into the repo root.

Prerequisites:
  1. Request access: https://huggingface.co/facebook/sam3 (Meta license).
  2. Authenticate (pick one):
       set HF_TOKEN=hf_your_token
       py -c "from huggingface_hub import login; login()"
     Or: .\\scripts\\download_sam3.ps1 -Token hf_xxxxx

Usage:
  pip install huggingface_hub
  py scripts/download_sam3.py
  .\\scripts\\download_sam3.ps1
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "sam3.pt"
REPO_ID = "facebook/sam3"
FILENAME = "sam3.pt"


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    print(f"[INFO] Downloading {REPO_ID} / {FILENAME} …")
    print("[INFO] You must have approved access on Hugging Face first.")
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=str(ROOT),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        print(f"[ERROR] Download failed: {exc}")
        print(
            "\nManual steps:\n"
            "  1. https://huggingface.co/facebook/sam3 - request access\n"
            "  2. Download sam3.pt from the Files tab\n"
            f"  3. Copy to: {DEST}\n"
            "  4. Or set HF_TOKEN and re-run this script\n"
        )
        sys.exit(1)

    downloaded = Path(path)
    if downloaded.resolve() != DEST.resolve() and downloaded.is_file():
        import shutil
        shutil.copy2(downloaded, DEST)
    if DEST.is_file():
        mb = DEST.stat().st_size / (1024 * 1024)
        print(f"[OK] {DEST} ({mb:.1f} MB)")
    else:
        print(f"[WARN] Check for weights under: {ROOT}")


if __name__ == "__main__":
    main()
