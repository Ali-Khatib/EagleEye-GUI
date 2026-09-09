# Folder layout

Repository root: `SAHI/` (this project).

```
SAHI/
├── README.md                 # Quick start (points here for full report)
├── requirements.txt          # Python deps (YOLO, SAHI, SAM2, OpenCV, …)
├── .gitignore
│
├── docs/                     # Documentation
│   ├── PROJECT_REPORT.md     # Full project explanation (read this for defense/presentation)
│   └── FOLDER_LAYOUT.md      # This file
│
├── core/                     # Shared Python libraries
│   ├── paths.py              # Central path constants
│   ├── sam2_support.py       # SAM 2.1 load, dataset checkpoints, mask generator
│   └── experiment_utils.py   # GT labels, IoU metrics, mask/box helpers
│
├── experiments/              # Seven pipeline scripts + validation
│   ├── _bootstrap.py         # Adds repo root to sys.path; OUTPUT_DIR, DEMO paths
│   ├── exp1_yolo_only.py … exp6_sahi_yolo_sam.py
│   ├── run_validation.py     # Official full-val YOLO metrics (paper numbers)
│   ├── main.py               # Legacy one-off SAHI+SAM CLI demo
│   └── videowithlivegui.py   # Legacy live video (SAM v1, not SAM2)
│
├── scripts/
│   ├── data/                 # Dataset preparation
│   │   ├── visdrone_to_yolo.py
│   │   ├── prepare_yolo_kitti.py
│   │   └── train_yolo.py
│   └── training/             # SAM 2.1 pseudo-masks + fine-tune
│       ├── generate_sam_pseudo_masks.py
│       ├── finetune_sam2.py
│       ├── build_sam2_resume_from_last.py
│       ├── run_sam2_finetune_visdrone.ps1
│       └── run_sam2_finetune_kitti.ps1
│
├── data/
│   └── demo/                 # Single test images + YOLO labels for web UI / exps
│       ├── test_image_visdrone.jpg / .txt
│       ├── test_image_kitti.jpg / .txt
│       └── test_image_stock.jpg / .txt
│
├── dataset/                  # VisDrone2019-DET (large — keep on disk here)
│   ├── VisDrone.yaml
│   └── VisDrone2019-DET-{train,val,test-dev}/…
│
├── kitti/                    # KITTI object detection (YOLO layout)
│   ├── kitti.yaml
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── sam2_pseudo_masks/      # After pseudo-mask generation
│
├── models/                   # Base SAM 2.1 weights (gitignored *.pt)
│   └── sam2/sam2.1_hiera_large.pt
│
├── runs/                     # Training outputs (gitignored)
│   ├── detect/train9/…       # VisDrone YOLO best.pt
│   ├── detect/train2/…       # KITTI YOLO best.pt
│   └── sam2_finetune/{visdrone,kitti}/best.pt
│
├── outputs/
│   ├── experiments/          # Per-pipeline demo runs (CSV, TXT, JPG)
│   └── validation/           # Full-dataset YOLO validation (paper metrics)
│
├── webapp/                   # FastAPI + React comparison UI
│   ├── start.ps1
│   ├── backend/main.py
│   └── frontend/
│
├── yolo11n.pt                # Stock COCO baseline weights
└── yolo26n.pt
```

## Why large data stayed in place

Moving `dataset/` (VisDrone) or `kitti/images/` would require re-downloading or
updating every absolute path in YAML configs. Those folders stay at the paths
already referenced in `dataset/VisDrone.yaml` and `kitti/kitti.yaml`.

## Run commands (after reorg)

From repo root:

```powershell
python experiments/exp1_yolo_only.py visdrone
python experiments/run_validation.py --dataset visdrone
python scripts/training/finetune_sam2.py --data dataset/VisDrone.yaml ...
.\scripts\training\run_sam2_finetune_visdrone.ps1 -SkipPseudo
.\webapp\start.ps1
```
