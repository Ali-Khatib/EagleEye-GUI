#!/usr/bin/env python3
"""Load one SAM3 training batch and run forward+backward (VRAM/crash debug)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAM3_DIR = ROOT / "external" / "sam3"
sys.path.insert(0, str(SAM3_DIR))
os.chdir(SAM3_DIR)

os.environ.setdefault("USE_PERFLIB", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam3.model.utils.misc import copy_data_to_device
from sam3.train.utils.train_utils import get_amp_type, register_omegaconf_resolvers

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/eagleeye/eagleeye_visdrone_smoke.yaml"


def main() -> None:
    register_omegaconf_resolvers()
    with initialize_config_module(config_module="sam3.train", version_base="1.2"):
        cfg = compose(config_name=CONFIG)
    OmegaConf.resolve(cfg)

    print("[debug] Building model...")
    model = instantiate(cfg.trainer.model)
    model = model.cuda().train()
    bb = model.backbone
    bb.act_ckpt_whole_vision_backbone = True
    bb.act_ckpt_whole_language_backbone = True
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[debug] Trainable params: {trainable / 1e6:.1f}M")

    print("[debug] Building dataset + batch...")
    dataset = instantiate(cfg.trainer.data.train.dataset)
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=instantiate(cfg.trainer.data.train.collate_fn),
    )
    batch = next(iter(loader))
    key, batch = batch.popitem()
    batch = copy_data_to_device(batch, "cuda", non_blocking=True)

    loss_mod = instantiate(cfg.trainer.loss.all)
    loss_mod = loss_mod.cuda().train()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )

    print("[debug] Forward...")
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        find_stages = model(batch)
        find_targets = [model.back_convert(x) for x in batch.find_targets]
        losses = loss_mod(find_stages, find_targets)
        if isinstance(losses, dict):
            loss = losses.get("core_loss", next(iter(losses.values())))
        else:
            loss = losses

    print(f"[debug] Loss: {float(loss):.4f}")
    print("[debug] Backward...")
    loss.backward()
    optim.step()
    print("[debug] OK — one step completed")


if __name__ == "__main__":
    main()
