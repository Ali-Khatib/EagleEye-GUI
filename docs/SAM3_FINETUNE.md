# SAM3 fine-tuning (Meta official training)

Ultralytics in this repo is **inference-only** for SAM3. Fine-tuning uses Meta's
[facebookresearch/sam3](https://github.com/facebookresearch/sam3) training code.

## Quick overview

| Step | What |
|------|------|
| 1 | Convert YOLO labels → COCO (pseudo rectangle masks) |
| 2 | Clone Meta `sam3` repo + `pip install -e ".[train]"` |
| 3 | Train (~10–15 epochs, RTX 4060, many hours) |
| 4 | Export checkpoint → `runs/sam3_finetune/{dataset}/sam3.pt` |
| 5 | Re-run SAM3 benchmark pipelines (auto-loads fine-tuned weights) |

**Note:** Meta SAM3 training is tested on **Linux**. On Windows, use **WSL2** or a Linux machine.

## Step 1 — Prepare COCO data

```powershell
# Full VisDrone train/val
python scripts/training/prepare_sam3_coco.py --dataset visdrone

# Smoke test (200 images per split)
python scripts/training/prepare_sam3_coco.py --dataset visdrone --limit 200

# KITTI
python scripts/training/prepare_sam3_coco.py --dataset kitti
```

Output: `data/sam3_coco/{dataset}/train/` and `val/` with `_annotations.coco.json`.

Masks are **rectangle polygons from YOLO boxes** (same idea as the old SAM2 pseudo-mask workflow).

## Step 2 — Install Meta SAM3 training (WSL/Linux recommended)

```bash
cd /path/to/SAHI
git clone https://github.com/facebookresearch/sam3.git external/sam3
cd external/sam3
pip install -e ".[train]"
```

You also need the base `sam3.pt` from HuggingFace (same file already at repo root).

## Step 3 — Generate config + train

```powershell
python scripts/training/generate_sam3_train_config.py --dataset visdrone --epochs 15 --gpus 1
```

Then in WSL/Linux from `external/sam3`:

```bash
python -m sam3.train.train \
  -c /mnt/c/Users/khati/PycharmProjects/SAHI/runs/sam3_finetune/visdrone/eagleeye_visdrone_ft.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

Checkpoints save to: `runs/sam3_finetune/visdrone/checkpoints/checkpoint.pt`

RTX 4060 tips (in config):
- `train_batch_size: 1`
- `gradient_accumulation_steps: 4`
- `resolution: 1008`
- Start with `--limit 200` on prepare script for a dry run

## Step 4 — Export for our pipelines

```powershell
python scripts/training/export_sam3_finetune.py --dataset visdrone
```

Copies checkpoint → `runs/sam3_finetune/visdrone/sam3.pt`

## Step 5 — Re-run SAM3 benchmark pipelines

Fine-tuned weights are picked **automatically** when the dataset is VisDrone/KITTI.

```powershell
# VisDrone — sam3_only, sam3_yolo, yolo_sahi_sam3
.\scripts\run_visdrone_sam_pipelines.ps1

# Or one pipeline smoke test
python pipelines/compare_all.py ... --modes sam3_yolo --skip-existing
```

To force base (non fine-tuned) weights: set env `SAM3_WEIGHTS=C:\...\SAHI\sam3.pt`

## Paper wording

> YOLO detectors were fine-tuned per dataset (VisDrone: 100 epochs; KITTI: 50 epochs).
> SAM3 was fine-tuned for 15 epochs on pseudo-instance masks derived from training-set
> bounding boxes using Meta's official training code. All pipelines were re-evaluated on
> held-out validation / test-dev splits.

## What to expect

- Fine-tuning may **reduce false positives** from text prompts on your domain
- **Box F1** may still lose to YOLO-only on KITTI — that's a valid finding
- Biggest gains likely on **VisDrone** (aerial) where pretrained SAM3 struggles most
- Compare **before vs after** fine-tune tables in the paper (strong ablation section)
