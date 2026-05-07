# SAHI + YOLO Live Video Detection

Real-time object detection on video using **YOLO** and **SAHI** (Slicing Aided Hyper Inference), with a live OpenCV GUI for toggling settings on the fly.
Includes an image pipeline that combines **SAHI + SAM** for pixel-level masks.

## Features

- **Two detection modes** (toggle with `M`):
  - **YOLO** — fast, full-frame inference
  - **SAHI** — sliced inference for better small-object detection
- **SAM segmentation (image demo)** — SAM refines masks from detection boxes (SAM does not detect by itself)
- **Three confidence presets** (keys `1` / `2` / `3`)
- **Temporal filtering** (toggle with `T`) — suppresses flickering detections
- **Live HUD overlay** showing current settings and detection stats
- Saves annotated output video

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the live detection GUI

Place your video at `demo_data/test.mp4` (or edit the path in `videowithlivegui.py`), then:

```bash
python videowithlivegui.py
```

### 3. Run the SAHI + SAM image demo

```bash
python main.py --mode sahi
```

### 4. Run the YOLO + SAM image demo (no slicing)

```bash
python main.py --mode yolo
```

## Keyboard Controls (Live GUI)

| Key | Action |
|-----|--------|
| `1` | High confidence threshold (0.35) |
| `2` | Mid confidence threshold (0.10) |
| `3` | Low confidence threshold (0.05) |
| `T` | Toggle temporal filtering |
| `M` | Toggle YOLO ↔ SAHI mode |
| `Q` / `Esc` | Quit |

## Requirements

- Python 3.10+
- `opencv-python`
- `ultralytics` (YOLOv8/v11)
- `sahi`

Model weights (`yolo11x.pt`, `sam2.x.pt`) are downloaded automatically on first run.

## Project Structure

```
├── main.py                  # YOLO/SAHI + SAM image demo
├── sahi_sam_runner.py        # Small SAHI+SAM runner (no window)
├── videowithlivegui.py      # Live video detection with GUI
├── requirements.txt
├── PROJECT_EXPLANATION.txt  # Detailed ELI5-style explanation
└── demo_data/
    ├── small-vehicles1.jpeg # Sample test image
    ├── terrain2.png         # Sample test image
    └── prediction_visual.png
```

## License

MIT

