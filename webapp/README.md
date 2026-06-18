# EagleEye AI — Web UI

Local web interface for the 6 YOLO / SAHI / SAM 3 pipelines.

- **Frontend**: Vite + React + TypeScript + Tailwind (port `5173`)
- **Backend**: FastAPI + Uvicorn (port `8000`)

Everything is local — `127.0.0.1` only.

## Quick start

```powershell
.\webapp\start.ps1
```

## Manual start

**Backend:**

```powershell
cd webapp/backend
pip install -r requirements.txt
python main.py
```

**Frontend:**

```powershell
cd webapp/frontend
npm install
npm run dev
```

Open http://localhost:5173

## Tabs

- **Home** — project cover and intro
- **Pipelines** — 6 cards; click Run to execute a pipeline
- **Results** — leaderboard, charts, image preview

## Datasets

Toggle **VisDrone / KITTI / Stock (COCO)** in the top bar. Each dataset uses its own YOLO checkpoint and keeps separate outputs under `outputs/webapp/<dataset>/`.

Demo images live in `data/demo/`. Upload through the UI or replace those files.

## Model files required

- `sam3.pt` (repo root) — download via `python scripts/download_sam3.py`
- `runs/detect/train9/weights/best.pt` — VisDrone YOLO
- `runs/detect/train2/weights/best.pt` — KITTI YOLO
- `runs/detect/train_v8_visdrone/weights/best.pt` — pipeline 6
- `yolo11n.pt` — Stock baseline

## Troubleshooting

HTTP 500 in the browser usually means the FastAPI backend is not running. Start `python main.py` in `webapp/backend` and refresh.

Long SAM / SAHI runs can take several minutes. API timeout is 15 minutes per run.
