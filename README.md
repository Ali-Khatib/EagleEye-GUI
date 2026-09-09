# EagleEye AI

Local research workspace for comparing **five** computer-vision pipelines on **VisDrone**, **KITTI**, and a **COCO stock** baseline:

| Mode | Pipeline |
|------|----------|
| `yolo_only` | YOLO only |
| `yolo_sahi` | YOLO + SAHI |
| `sam3_only` | SAM 3 only (text prompts) |
| `sam3_yolo` | SAM 3 + YOLO |
| `yolo_sahi_sam3` | **YOLO + SAHI + SAM 3** (main pipeline) |

**Full report:** [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)  
**Folder map:** [docs/FOLDER_LAYOUT.md](docs/FOLDER_LAYOUT.md)

## Why SAM 3 replaces SAM 2.1

SAM 2.1 was a strong mask refiner but required separate Meta `sam2` installs, AMG heuristics, and dataset-specific fine-tuning. **SAM 3** (Meta, via Ultralytics) adds **open-vocabulary concept segmentation** with text prompts and box prompts on a single `sam3.pt` checkpoint. This project uses SAM 3 to:

- Segment by class name (`person`, `car`, `truck`, …) in `sam3_only`
- Refine YOLO/SAHI boxes into high-quality masks without inventing extra detections (`sam3_yolo`, `yolo_sahi_sam3`)

## Main pipeline: YOLO + SAHI + SAM 3

1. **YOLO** — fine-tuned detector (`runs/detect/train9/weights/best.pt` for VisDrone).
2. **SAHI** — 256×256 slices, 50% overlap, conf 0.12, GREEDYNMM merge, YOLO imgsz 640 per tile (microscopic/small-object tuned). Override via env: `SAHI_SLICE`, `SAHI_OVERLAP`, `SAHI_CONF`.
3. **SAM 3** — box-prompt mask refinement per detection; **class IDs stay from YOLO/SAHI**.

## Install

```bash
pip install -r requirements.txt
```

### SAM 3 weights

1. Accept the license on [Hugging Face — facebook/sam3](https://huggingface.co/facebook/sam3).
2. Download `sam3.pt` and place it at the repo root, or set:

```bash
set SAM3_WEIGHTS=C:\path\to\sam3.pt
```

Ultralytics will also fetch CLIP assets on first SAM 3 run if needed.

## Compare all modes (fair evaluation)

```powershell
python compare_all.py ^
  --source kitti/images/val ^
  --ground-truth kitti/labels/val ^
  --weights runs/detect/train9/weights/best.pt ^
  --classes person pedestrian car van truck bus bicycle motorcycle ^
  --output outputs ^
  --limit 50
```

Or:

```powershell
python pipelines/compare_all.py --source ... --ground-truth ... --weights ...
```

### Outputs

```
outputs/
  yolo_only/          # images/, labels/, masks/, metrics.json, summary.csv, vis/
  yolo_sahi/
  sam3_only/
  sam3_yolo/
  yolo_sahi_sam3/
  comparison_results.csv
  recommendation.txt
  best_mode.txt
```

Metrics are computed from predictions vs ground truth (no hardcoded scores): mAP50, precision, recall, F1, small-object recall, mask quality proxy, FPS, false positives, missed objects, duplicate count.

`recommendation.txt` explains trade-offs (e.g. `yolo_only` is faster; `sam3_only` is less controlled for box mAP; `yolo_sahi_sam3` when small-object recall and mask quality lead).

## Web app

```powershell
.\webapp\start.ps1
```

Open http://localhost:5173 — run any of the five pipelines on the demo image.

## Run one pipeline (demo image)

```powershell
python -m experiments.exp1_yolo_only visdrone
python -m experiments.exp5_yolo_sahi_sam3 kitti
```

## Official YOLO validation (paper)

```powershell
python experiments/run_validation.py --dataset visdrone
```

## Project layout

```
core/              # sam3_support, sahi_config, compare_metrics, detection_ops
pipelines/         # yolo_only, yolo_sahi, sam3_*, compare_all.py
experiments/       # exp1–exp5 demo runners
scripts/data/      # dataset prep, YOLO training
data/demo/         # test images for UI / CLI
outputs/           # comparison + experiment results
webapp/            # FastAPI + React UI
```

## Interpreting results

- **yolo_only** — fastest; good baseline box AP; weak on tiny/distant objects.
- **yolo_sahi** — better small-object recall; more duplicates possible without NMS tuning.
- **sam3_only** — strong segmentation from text; class-agnostic box metrics unless prompts match GT classes.
- **sam3_yolo** — controlled detection + mask quality; YOLO limits what SAM 3 can segment.
- **yolo_sahi_sam3** — recommended for hard scenes: SAHI finds candidates, SAM 3 polishes masks.
