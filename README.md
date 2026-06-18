# EagleEye AI — Web UI & Pipelines

Local web app for running and comparing **six** computer-vision pipelines on a single image:

| Pipeline | What it does |
|----------|----------------|
| YOLO only | Full-frame object detection |
| YOLO + SAHI | Tiled detection for small / distant objects |
| SAM 3 only | Text-prompt segmentation (open vocabulary) |
| SAM 3 + YOLO | YOLO boxes refined with SAM 3 masks |
| YOLO + SAHI + SAM 3 | SAHI tiled detection → SAM 3 mask refinement |
| YOLOv8 + SAHI + SAM 3 | Same as above with YOLOv8 weights (VisDrone) |

Supports three dataset presets: **VisDrone**, **KITTI**, and **Stock (COCO)**. Upload your own image or use the bundled demo photos.

Everything runs locally on `127.0.0.1` — no data leaves your machine.

## Quick start

```powershell
# 1. Python deps (from repo root)
pip install -r requirements.txt
pip install -r webapp/backend/requirements.txt

# 2. Model weights (see below)

# 3. Launch UI (backend + frontend)
.\webapp\start.ps1
```

Open **http://localhost:5173**

## What the website does

```
Browser (5173)  →  Vite proxy  →  FastAPI (8000)  →  python experiments/expN_*.py
                                                  →  outputs/webapp/<dataset>/<pipeline>/
```

- **Home** — project overview
- **Pipelines** — run any of the 6 pipelines on the current image; outputs saved per dataset
- **Results** — leaderboard (F1, precision, recall, FPS, runtime), charts, before/after preview

Switch the **VisDrone / KITTI / Stock** toggle in the top bar to keep separate result sets per dataset.

## Model weights (not in git)

Place these files before running pipelines that need them:

| File | Used for |
|------|----------|
| `sam3.pt` | All SAM 3 pipelines |
| `runs/detect/train9/weights/best.pt` | VisDrone YOLO |
| `runs/detect/train2/weights/best.pt` | KITTI YOLO |
| `runs/detect/train_v8_visdrone/weights/best.pt` | Pipeline 6 (YOLOv8) |
| `yolo11n.pt` | Stock (COCO) baseline |

Download SAM 3:

```powershell
pip install huggingface_hub
# Accept license at https://huggingface.co/facebook/sam3 first
python scripts/download_sam3.py
```

Or set `SAM3_WEIGHTS=C:\path\to\sam3.pt`.

## Run a pipeline from the CLI (no UI)

```powershell
python experiments/exp1_yolo_only.py visdrone
python experiments/exp5_yolo_sahi_sam3.py kitti
```

## Repo layout

```
core/           Shared inference logic (SAHI, SAM 3, YOLO, metrics)
pipelines/      Five pipeline implementations
experiments/    Thin CLI wrappers the UI subprocesses
webapp/         React frontend + FastAPI backend
data/demo/      Sample images + YOLO label files
outputs/        Generated results (gitignored contents)
scripts/        Utility to download sam3.pt
```

## Manual start (two terminals)

**Backend** (port 8000):

```powershell
cd webapp/backend
python main.py
```

**Frontend** (port 5173):

```powershell
cd webapp/frontend
npm install
npm run dev
```

## Requirements

- Python 3.10+
- Node.js 18+ (for the frontend)
- CUDA GPU recommended (CPU works but slow for SAM 3)

Python packages: OpenCV, Ultralytics (YOLO + SAM 3), SAHI, PyTorch, FastAPI.
