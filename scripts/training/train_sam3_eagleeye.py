#!/usr/bin/env python3
"""SAM3 train entry with EagleEye tweaks for RTX 4060 8GB on Windows."""
from __future__ import annotations

import os
import runpy
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
SAM3_DIR = ROOT / "external" / "sam3"
if str(SAM3_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_DIR))

os.environ.setdefault("USE_PERFLIB", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from sam3.train.trainer import Trainer  # noqa: E402
from sam3.train.utils import train_utils  # noqa: E402


def _patch_single_gpu_windows() -> None:
    """Single-GPU Windows: init gloo for loss all_reduce, skip DDP wrap."""
    def _setup_dist(backend, timeout_mins):
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not dist.is_initialized():
            win_backend = "gloo"
            print(f"[EagleEye] Single-GPU — init {win_backend} (world_size=1)")
            dist.init_process_group(
                backend=win_backend,
                rank=0,
                world_size=1,
                timeout=timedelta(minutes=timeout_mins),
            )
        return dist.get_rank()

    train_utils.setup_distributed_backend = _setup_dist

    def _skip_ddp(self, distributed_conf, accelerator):
        print("[EagleEye] Single-GPU mode — model stays unwrapped (no DDP)")

    Trainer._setup_ddp_distributed_training = _skip_ddp


_patch_single_gpu_windows()


def _unwrap(trainer: Trainer):
    model = trainer.model
    if hasattr(model, "module"):
        model = model.module
    return model


def _freeze_backbone(trainer: Trainer) -> None:
    model = _unwrap(trainer)
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            param.requires_grad = False
            frozen += param.numel()
        else:
            trainable += param.numel()
    print(f"[EagleEye] Frozen backbone params: {frozen / 1e6:.1f}M")
    print(f"[EagleEye] Trainable params: {trainable / 1e6:.1f}M")


def _enable_low_vram_mode(trainer: Trainer) -> None:
    """Activation checkpointing on frozen backbones — big VRAM win on 8GB."""
    model = _unwrap(trainer)
    bb = model.backbone
    bb.act_ckpt_whole_vision_backbone = True
    bb.act_ckpt_whole_language_backbone = True
    print("[EagleEye] Backbone activation checkpointing: ON")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_orig_setup = Trainer._setup_components


def _setup_components_with_freeze(self):
    _orig_setup(self)
    _freeze_backbone(self)
    _enable_low_vram_mode(self)


Trainer._setup_components = _setup_components_with_freeze

if __name__ == "__main__":
    os.chdir(SAM3_DIR)
    runpy.run_module("sam3.train.train", run_name="__main__")
