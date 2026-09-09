# EagleEye AI — Web UI

Local web interface for the 6 SAHI / YOLO / SAM experiments.

- **Frontend**: Vite + React + TypeScript + Tailwind (port `5173`)
- **Backend**: FastAPI + Uvicorn (port `8000`)

Everything is local — `127.0.0.1` only. No network calls leave the machine.
The backend's only job is to bridge the browser ↔ filesystem ↔ Python
subprocesses (a browser can't read your CSVs or run `python expN.py` by
itself, hence the small FastAPI layer).

```
Browser (5173)  →  Vite proxy  →  FastAPI (8000)  →  python expN_*.py
                                                  →  outputs/experiments/*.csv|jpg
```

## Quick start (one command)

```powershell
.\webapp\start.ps1
```

This spawns the backend in one window and the frontend in another, then
opens both. Press Ctrl+C in the original window to stop them. First run
will also do `npm install` automatically.

## Manual two-terminal start (if you prefer)

### 1. Start the backend (port 8000)

```powershell
cd webapp/backend
pip install -r requirements.txt
python main.py
```

Sanity check: open <http://127.0.0.1:8000/api/health>.

### 2. Start the frontend (port 5173)

```powershell
cd webapp/frontend
npm install
npm run dev
```

Then open <http://localhost:5173>. Vite proxies `/api/*` to FastAPI on
port 8000, so the only URL you visit is 5173.

## "Why are there 500 errors in the browser?"

Almost always: the FastAPI backend isn't running, so the Vite proxy can't
reach `127.0.0.1:8000` and surfaces the connection error as HTTP 500.
Fix: start `python main.py` (or use `start.ps1`) and refresh the page.

## 3. Use it

- **Home** — project cover (TÜBİTAK ARDEB 3501 · 124E099) and a quick tour.
- **Pipelines** — 6 cards. Click `Run` on any pipeline (or `Run all 6`) to
  execute the matching script. Outputs land in `outputs/experiments/`.
- **Results** — sortable leaderboard with F1 / precision / recall / accuracy /
  FPS / runtime / count / YOLO size MB / YOLO parameter count / SAM size MB,
  bar charts for quality + speed, and a before/after image preview for the
  selected pipeline.

## Datasets (VisDrone + KITTI)

The UI has a **VisDrone / KITTI / Stock (COCO)** toggle in the top bar. Switching it changes
every read/write — uploads, runs, image previews, and the leaderboard — so
you get separate result sets per pipeline. VisDrone uses `train9`, KITTI uses
`train2`, and Stock (COCO) uses the unfine-tuned `yolo11n.pt` baseline.

**Files per dataset**:

```
data/demo/test_image_visdrone.jpg       # input image
data/demo/test_image_visdrone.txt       # optional YOLO ground-truth labels
data/demo/test_image_kitti.jpg
data/demo/test_image_kitti.txt
data/demo/test_image_stock.jpg
data/demo/test_image_stock.txt
```

**Outputs per dataset** (suffixed):

```
outputs/experiments/01_yolo_only_visdrone.{jpg,csv,txt}
outputs/experiments/01_yolo_only_kitti.{jpg,csv,txt}
outputs/experiments/01_yolo_only_stock.{jpg,csv,txt}
...
outputs/experiments/06_sahi_yolo_sam_visdrone.{jpg,csv,txt}
```

**Migrating from the old single image**: each script falls back to
`data/demo/test_image.jpg` when the dataset-specific file is missing, so
your current setup keeps working until you upload dedicated KITTI / VisDrone
samples through the UI.

**Running scripts directly** (without the UI):

```powershell
python experiments/exp1_yolo_only.py visdrone
python experiments/exp1_yolo_only.py kitti
# or via env var
$env:EXP_DATASET = "kitti"; python experiments/exp3_sahi_yolo.py
```

## Metrics Policy

The website is automated and intentionally simple: give a picture, choose a
pipeline, and see visual results/FPS/runtime. If you pick a random validation
image, the backend loads its matching ground-truth label automatically.

Research-paper metrics do **not** come from one image in the website. They come
from official validation on the full validation split:

```powershell
python experiments/run_validation.py --dataset visdrone
python experiments/run_validation.py --dataset kitti
```

## Dataset → YAML → YOLO checkpoint map

The 6 inference scripts only load the `.pt` weights, but the project also
ships the dataset YAML files used to train each one. Keep them aligned:

| Dataset  | Dataset YAML              | YOLO checkpoint                          | Classes |
|----------|---------------------------|------------------------------------------|---------|
| VisDrone | `dataset/VisDrone.yaml`   | `runs/detect/train9/weights/best.pt`     | 10 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor) |
| KITTI    | `kitti/kitti.yaml`        | `runs/detect/train2/weights/best.pt`     | 8 (car, van, truck, pedestrian, person_sitting, cyclist, tram, misc) |
| Stock    | n/a                       | `yolo11n.pt`                             | COCO classes, no project fine-tuning |

The same map is hard-coded at the top of every `expN_*.py`:

```python
YOLO_MODELS = {
    "visdrone": "runs/detect/train9/weights/best.pt",
    "kitti":    "runs/detect/train2/weights/best.pt",
    "stock":    "yolo11n.pt",
}
```

If a future training run lands in a different folder (e.g. `train10`),
update this dictionary in all 6 scripts.

## Notes

- Long-running pipelines (especially SAM and SAHI+SAM) can take minutes.
  The API timeout is 15 minutes per run.
- `sam_checkpoint_size_mb` reports `N/A` for pipelines that don't actually
  use SAM (YOLO only, SAHI+YOLO). Likewise `yolo_model_size_mb` and
  `yolo_parameter_count` are `N/A` for SAM-only and SAHI+SAM. This was
  fixed so the comparison table is honest.
- Make sure these model files exist before running:
  - `runs/detect/train9/weights/best.pt`  (VisDrone YOLO)
  - `runs/detect/train2/weights/best.pt`  (KITTI YOLO)
  - `models/sam2/sam2.1_hiera_tiny.pt`    (shared SAM 2.1 checkpoint)

## SAM Metric Interpretation

SAM-only and SAHI+SAM do not predict object classes. Their metrics are
class-agnostic IoU approximations, not normal class-based detection metrics.
The Results tab separates them under **Segmentation-only / class-agnostic
results** so they are not ranked against YOLO detection pipelines.

YOLO + SAM and SAHI + YOLO + SAM use SAM-refined tight boxes from the SAM 2.1
masks. SAM can tighten detector boxes and drop boxes where SAM produced no
usable mask, but SAM still does not predict object classes.

FPS and runtime remain valid for every mode.

## Official YOLO Validation

For paper-quality dataset-level YOLO metrics, use official Ultralytics
validation instead of single-image dashboard metrics:

```powershell
python experiments/run_validation.py --dataset visdrone
python experiments/run_validation.py --dataset kitti
```

This runs official Ultralytics validation and saves Precision, Recall, F1,
mAP50, mAP50-95, confusion matrix, PR curve, predictions, and plots under
`outputs/validation/`. Treat dashboard metrics as demo/comparison metrics only.

## SAM 2.1 Setup

Old `segment_anything`, `sam_model_registry`, `SamPredictor`,
`SamAutomaticMaskGenerator`, and `models/sam_vit_b.pth` are no longer used by
the six experiment files. The SAM modes now use Meta SAM 2.1:

```powershell
pip uninstall segment-anything -y
pip uninstall sam2 -y
pip install git+https://github.com/facebookresearch/sam2.git
```

Checkpoint expected by the scripts:

```text
models/sam2/sam2.1_hiera_tiny.pt
```

Old SAM cleanup has already been completed for this workspace.
