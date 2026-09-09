# SAHI / YOLO / SAM 2.1 — Complete Project Report

**Project:** TÜBİTAK ARDEB 3501 — *124E099*  
**Theme:** Real-time scalable AI camera design for low-cost edge devices  
**Goal:** Compare how **YOLO** (fast detectors), **SAHI** (sliced inference for small objects), and **SAM 2.1** (segmentation) combine on **VisDrone** (aerial) and **KITTI** (road) imagery.

This document is written so you can explain every sector of the repository: research idea, folder layout, each pipeline, training, metrics, web app, and design choices.

---

## 1. Research question (what & why)

### What we built

A **controlled comparison** of six end-to-end computer-vision pipelines on the same images and ground truth:

| ID | Script | Idea in one sentence |
|----|--------|----------------------|
| **1** | `exp1_yolo_only.py` | Standard YOLO on full image — baseline speed/accuracy. |
| **2** | `exp2_sam_only.py` | SAM 2.1 Automatic Mask Generator (AMG) — segmentation without classes. |
| **3** | `exp3_sahi_yolo.py` | SAHI slices image → YOLO per tile → merge boxes (small objects). |
| **4** | `exp4_sahi_sam.py` | SAHI slices → SAM AMG per tile → merge masks (no YOLO). |
| **5** | `exp5_yolo_sam.py` | YOLO proposes boxes → SAM refines each box to a mask. |
| **6** | `exp6_sahi_yolo_sam.py` | SAHI+YOLO detections → SAM refines each detection mask. |

### Why these three technologies

- **YOLO (Ultralytics):** Single-shot object detection; fast; outputs class + box + confidence. Fine-tuned per dataset (`runs/detect/.../best.pt`).
- **SAHI:** Splits large / high-resolution images into overlapping crops, runs the detector on each crop, then **merges** detections with NMS. Critical for **small objects** in drone views (VisDrone).
- **SAM 2.1 (Meta):** Promptable segmenter; **AMG** can propose many masks without boxes; **box prompts** tighten masks around YOLO boxes. We **fine-tune only the mask decoder** on pseudo-masks so masks fit domain imagery.

### Datasets

| Dataset | Domain | Classes (examples) | YOLO config | SAM fine-tune |
|---------|--------|-------------------|-------------|---------------|
| **VisDrone** | UAV / crowded scenes | pedestrian, car, van, … | `dataset/VisDrone.yaml` | `runs/sam2_finetune/visdrone/best.pt` |
| **KITTI** | Driving / road | Car, Pedestrian, Cyclist, … | `kitti/kitti.yaml` | `runs/sam2_finetune/kitti/best.pt` |
| **Stock** | COCO baseline | 80 COCO classes | `yolo11n.pt` | Generic `models/sam2/sam2.1_hiera_large.pt` |

Ground truth in experiments uses **YOLO-format `.txt` labels** (normalized `class cx cy w h`). SAM-only pipelines use **class-agnostic** matching (IoU on boxes derived from masks).

---

## 2. Repository layout (where everything lives)

See **[FOLDER_LAYOUT.md](FOLDER_LAYOUT.md)** for the full tree. Summary:

| Folder | Role |
|--------|------|
| `core/` | Shared SAM loading, paths, metrics helpers |
| `experiments/` | All six pipelines + `run_validation.py` |
| `scripts/data/` | VisDrone→YOLO conversion, KITTI prep, YOLO training |
| `scripts/training/` | Pseudo-masks, SAM fine-tune, PowerShell runners |
| `data/demo/` | Default test images + labels for UI and CLI |
| `dataset/`, `kitti/` | Full datasets (not moved — paths in YAML) |
| `outputs/experiments/` | Per-run JPG/CSV/TXT from demos |
| `outputs/validation/` | Full validation metrics (paper-style) |
| `runs/` | YOLO and SAM training checkpoints (gitignored) |
| `webapp/` | FastAPI backend + React frontend |
| `docs/` | This report + folder map |

**Run everything from the repo root**, e.g.:

```powershell
python experiments/exp1_yolo_only.py visdrone
.\webapp\start.ps1
.\scripts\training\run_sam2_finetune_visdrone.ps1 -SkipPseudo
```

---

## 3. The six pipelines (how each works)

### Exp 1 — YOLO only

1. Load image (demo path or argv dataset: `visdrone` / `kitti` / `stock`).
2. `YOLO.predict()` on full frame.
3. Draw boxes; optional GT match → precision, recall, F1.
4. Write `outputs/experiments/01_yolo_only_{dataset}.{jpg,csv,txt}`.

**Use when explaining:** fastest detector baseline; no segmentation.

### Exp 2 — SAM only

1. Build SAM 2.1 (dataset-specific `best.pt` if present).
2. **Automatic Mask Generator** on full image → many mask proposals.
3. Draw masks + mask bounding boxes; GT evaluation is **class-agnostic** (SAM has no class head).

**Use when explaining:** pure segmentation / proposal quality; not a classifier.

### Exp 3 — SAHI + YOLO

1. `slice_image()` (SAHI) with overlap.
2. YOLO on each slice; merge with SAHI postprocess (NMS).
3. Class-aware metrics vs YOLO GT.

**Use when explaining:** small-object detection in large frames (VisDrone).

### Exp 4 — SAHI + SAM

1. Slice image like exp3.
2. SAM AMG per slice; deduplicate masks by IoU across slices.
3. Class-agnostic GT metrics.

**Use when explaining:** segmentation without a detector; expensive but finds regions YOLO might miss.

### Exp 5 — YOLO + SAM (refine)

1. YOLO full-image detections.
2. For each box, **SAM box prompt** → instance mask.
3. Metrics on **boxes** (same as YOLO stage); masks are visual refinement.

**Use when explaining:** classic “detect then segment”; benefits most from **fine-tuned SAM**.

### Exp 6 — SAHI + YOLO + SAM

1. SAHI + YOLO (like exp3).
2. SAM refines each merged detection.
3. Combines small-object detection with mask quality.

**Use when explaining:** full stack for hardest dataset (VisDrone).

---

## 4. Core code (`core/`)

### `sam2_support.py`

- Default model: **SAM 2.1 Hiera-Large** (`models/sam2/sam2.1_hiera_large.pt`).
- `resolve_sam2_paths(dataset)` → VisDrone/KITTI use `runs/sam2_finetune/{dataset}/best.pt` when the file exists; else warn and fall back to Large.
- Helpers: `build_sam2_model`, `create_sam2_predictor`, `create_automatic_mask_generator`.

### `experiment_utils.py`

- Load YOLO GT labels; convert to xyxy.
- IoU, detection matching, `evaluate_with_ground_truth`.
- Shared drawing / metric patterns used by multiple experiments.

### `paths.py`

- Central constants so scripts agree on `data/demo`, `outputs/experiments`, etc.

### `experiments/_bootstrap.py`

- Inserts repo root on `sys.path`.
- Defines `ROOT`, `DEMO_DIR`, `OUTPUT_DIR` for all `exp*.py` files.

---

## 5. Training stack (what you did)

### 5.1 YOLO fine-tuning

- **VisDrone:** labels under `dataset/VisDrone2019-DET-...`; trained weights `runs/detect/train9/weights/best.pt`.
- **KITTI:** prepared via `scripts/data/prepare_yolo_kitti.py`; weights `runs/detect/train2/weights/best.pt`.
- Training entry: `scripts/data/train_yolo.py` (Ultralytics API).

### 5.2 SAM 2.1 fine-tuning (pseudo-mask workflow)

**Problem:** KITTI/VisDrone provide **boxes**, not instance masks. SAM training needs masks.

**Solution (standard practice):**

1. **Pseudo-masks** — `scripts/training/generate_sam_pseudo_masks.py`  
   For each GT box, run frozen SAM 2.1 Large → save `.npz` per image under `{dataset}/sam2_pseudo_masks/{train|val}/`.

2. **Decoder-only fine-tune** — `scripts/training/finetune_sam2.py`  
   - Freeze image encoder; train **mask decoder** (~4M params).  
   - Loss: Meta recipe (focal + dice + IoU MSE).  
   - Logs `val_mean_iou` per epoch in `runs/sam2_finetune/{dataset}/log.tsv`.  
   - Saves `best.pt`, `last.pt`, optional `resume.pt`.

3. **Automation** — PowerShell:
   - `scripts/training/run_sam2_finetune_visdrone.ps1` (default `--max-long-edge 1024` for VRAM).
   - `scripts/training/run_sam2_finetune_kitti.ps1` (default full resolution; `-ContinueFromLast` rebuilds resume).

4. **Resume helper** — `build_sam2_resume_from_last.py` rebuilds `resume.pt` from `last.pt` + `log.tsv` after interrupted runs.

**Important:** VisDrone and KITTI val IoU are **not comparable** across datasets (resolution, object size, pseudo-mask difficulty).

### 5.3 What improved after fine-tune

- **exp5 / exp6:** tighter masks around YOLO boxes.  
- **exp2 / exp4:** better AMG / prompt behavior on domain texture.  
- Web app and `resolve_sam2_paths()` pick checkpoints automatically per dataset toggle.

---

## 6. Metrics (two levels — do not mix them up)

### A. Single-image experiments (`exp1`–`exp6`)

- One demo or uploaded image + optional `.txt` GT.
- Reports: precision, recall, F1, accuracy (and timing, model size, FPS where implemented).
- Outputs: `outputs/experiments/NN_*_{dataset}.jpg|csv|txt`.
- **Purpose:** qualitative comparison, live demos, TÜBİTAK progress visuals.

### B. Official validation (`experiments/run_validation.py`)

- Full **validation split** via Ultralytics `model.val()`.
- Metrics: mAP50, mAP50-95, Precision, Recall, F1 (dataset-level).
- Outputs: `outputs/validation/{dataset}/` (CSV, plots, confusion matrix).

**Rule for the paper:** cite **B** for quantitative claims; use **A** only as illustrative unless you extend validation to all pipelines equally.

---

## 7. Web application (`webapp/`)

### Architecture

- **Backend:** FastAPI (`webapp/backend/main.py`) — spawns `python experiments/expN_....py {dataset}` with `cwd=ROOT`.
- **Frontend:** React + Vite — dataset toggle, upload image, run pipeline, view image + metrics table.
- **Start:** `.\webapp\start.ps1` → backend :8000, frontend :5173.

### Backend responsibilities

- Registry of all **6 experiments** with human-readable names.
- Per-dataset paths: YOLO/SAM checkpoint readiness, demo images in `data/demo/`.
- Results served from `outputs/experiments/`.
- Random sample from val set (VisDrone/KITTI paths configured in backend).

### What to say in a demo

1. Pick **VisDrone** or **KITTI** (shows which weights load).  
2. Upload or use default test image.  
3. Run exp1 vs exp5 vs exp6 — compare boxes, masks, FPS in CSV.  
4. Mention stock/COCO for generic baseline without fine-tuning.

---

## 8. Technology stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| Detection | Ultralytics YOLO (v11 / v26 weights in repo) |
| Slicing | `sahi` library |
| Segmentation | Meta SAM 2.1 (`sam2` pip package) |
| Vision I/O | OpenCV, NumPy |
| API | FastAPI, Uvicorn |
| UI | React, TypeScript, Vite |
| Training logs | TSV (`log.tsv`), Ultralytics `runs/` |
| GPU | CUDA (e.g. RTX 4060 Laptop); scripts set `-Device cuda` |

---

## 9. Design decisions & ideas (talking points)

1. **Modular experiments** — Each pipeline is one script; shared logic in `core/`; easy to add exp8 without touching others.

2. **Dataset-aware weights** — One codebase; argv or UI switches VisDrone/KITTI/stock and loads the right YOLO + SAM checkpoints.

3. **Pseudo-mask fine-tune** — Avoids manual polygon annotation; leverages existing YOLO labels; decoder-only keeps VRAM and time manageable.

4. **SAHI only where needed** — exp1/2/5 skip slicing; exp3/4/6 pay extra compute for small objects.

5. **Fallback SAM** — If `best.pt` missing, warn once and use Hiera-Large so demos never hard-fail mid-review.

6. **Separated metrics** — Web/single-image for exploration; `run_validation.py` for paper-grade YOLO numbers.

7. **Folder reorg (2025)** — `core/`, `experiments/`, `scripts/`, `data/demo/`, `outputs/` so thesis/committee can navigate without a flat root of 20 scripts.

---

## 10. Typical workflows (cheat sheet)

### Run one experiment (CLI)

```powershell
cd C:\Users\khati\PycharmProjects\SAHI
python experiments/exp6_sahi_yolo_sam.py kitti
```

### Web UI

```powershell
.\webapp\start.ps1
```

### Paper metrics (YOLO validation only)

```powershell
python experiments/run_validation.py --dataset visdrone
python experiments/run_validation.py --dataset kitti
```

### Full SAM fine-tune (VisDrone)

```powershell
.\scripts\training\run_sam2_finetune_visdrone.ps1
# Continue epochs:
.\scripts\training\run_sam2_finetune_visdrone.ps1 -SkipPseudo -ContinueFromLast -TotalEpochs 15
```

### Prepare data (if rebuilding)

```powershell
python scripts/data/visdrone_to_yolo.py
python scripts/data/prepare_yolo_kitti.py
python scripts/data/train_yolo.py
```

---

## 11. Outputs you can show reviewers

| Artifact | Location | Meaning |
|----------|----------|---------|
| Annotated comparisons | `outputs/experiments/*.jpg` | Visual pipeline output |
| Per-run metrics | `outputs/experiments/*.csv` | FPS, precision, recall, … |
| SAM training curve | `runs/sam2_finetune/*/log.tsv` | `val_mean_iou` per epoch |
| YOLO training | `runs/detect/train*/` | Ultralytics curves |
| Validation report | `outputs/validation/` | mAP, PR curves |

---

## 12. Sector-by-sector explanation (presentation map)

| Sector | Explain using |
|--------|----------------|
| **Research goal** | Section 1, pipeline table |
| **VisDrone vs KITTI** | Section 1 datasets + Section 5.1 |
| **Why SAHI** | exp3/4/6, Section 3 |
| **Why SAM** | exp2/4/5/6, Section 5.2 |
| **Fine-tuning story** | Section 5.2, pseudo-masks |
| **Engineering** | Section 2, 4, 8 |
| **Demo** | Section 7, `webapp/` |
| **Scientific numbers** | Section 6B, `run_validation.py` |
| **Reproducibility** | Section 10, `FOLDER_LAYOUT.md` |

---

## 13. Known limitations (honest answers)

- **SAM is not trained for class labels** — class metrics in exp2/4 use agnostic matching.
- **Single-image metrics ≠ full mAP** — always pair demo numbers with `run_validation.py` for YOLO.
- **Compute:** SAHI + SAM on large images is slow; VisDrone fine-tune at 1024 long edge trades resolution for VRAM.
- **Stock dataset** — COCO weights; not domain-adapted; used as external baseline only.

---

## 14. File index (quick lookup)

| Need | File |
|------|------|
| YOLO baseline | `experiments/exp1_yolo_only.py` |
| SAM AMG | `experiments/exp2_sam_only.py` |
| SAHI+YOLO | `experiments/exp3_sahi_yolo.py` |
| SAHI+SAM | `experiments/exp4_sahi_sam.py` |
| YOLO→SAM | `experiments/exp5_yolo_sam.py` |
| SAHI+YOLO+SAM | `experiments/exp6_sahi_yolo_sam.py` |
| SAM loader | `core/sam2_support.py` |
| Metrics helpers | `core/experiment_utils.py` |
| Pseudo-masks | `scripts/training/generate_sam_pseudo_masks.py` |
| SAM train | `scripts/training/finetune_sam2.py` |
| API | `webapp/backend/main.py` |
| UI home | `webapp/frontend/src/components/HomeTab.tsx` |

---

*Last updated after repository reorganization into `core/`, `experiments/`, `scripts/`, `data/demo/`, and `outputs/`.*
