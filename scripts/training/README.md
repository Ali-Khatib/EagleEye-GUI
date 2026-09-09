# Training scripts

## YOLO (Ultralytics)

- Entry: `scripts/data/train_yolo.py`
- VisDrone: **100 epochs** → `runs/detect/train9/weights/best.pt`
- KITTI: **50 epochs** → `runs/detect/train2/weights/best.pt`

## SAM3 fine-tuning (Meta official — not Ultralytics)

This repo uses SAM3 for **inference** via Ultralytics. To **fine-tune** SAM3 on
VisDrone/KITTI, follow **[docs/SAM3_FINETUNE.md](../../docs/SAM3_FINETUNE.md)**.

Quick start:

```powershell
python scripts/training/prepare_sam3_coco.py --dataset visdrone
python scripts/training/generate_sam3_train_config.py --dataset visdrone --epochs 15
# Train in WSL/Linux with Meta sam3 repo (see doc)
python scripts/training/export_sam3_finetune.py --dataset visdrone
.\scripts\run_visdrone_sam_pipelines.ps1   # re-benchmark SAM3 pipelines
```

Fine-tuned weights: `runs/sam3_finetune/{visdrone|kitti}/sam3.pt` (auto-used by pipelines).
