import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from collections import deque

# 🔥 SAM IMPORTS
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

video_path = "demo_data/test2.mp4"

# SAM TOGGLE
USE_SAM = False

sam = None
predictor = None
SAM_EVERY_N_FRAMES = 15
SAM_MOVE_THRESHOLD = 40
last_sam_boxes = []
cached_sam_masks = []

def get_sam_predictor():
    global sam, predictor
    if predictor is None:
        sam_checkpoint = "models/sam_vit_b.pth"
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
    return predictor

# SETTINGS
CONFIDENCE_PRESETS      = [0.30, 0.10, 0.05]
PRESET_LABELS           = ["HIGH (0.35)", "MID  (0.10)", "LOW  (0.05)"]
confidence_preset_index = 1
CONFIDENCE_THRESHOLD    = CONFIDENCE_PRESETS[confidence_preset_index]
MODEL_CONFIDENCE_FLOOR  = 0.15

USE_SAHI             = False
TEMPORAL_ENABLED     = False
TEMPORAL_WINDOW      = 3
TEMPORAL_MIN_HITS    = 2
TEMPORAL_DIST_RATIO  = 0.75
recent_history       = deque(maxlen=TEMPORAL_WINDOW - 1)

SLICE_HEIGHT      = 684
SLICE_WIDTH       = 684
OVERLAP_H         = 0.25
OVERLAP_W         = 0.25
POSTPROCESS_TYPE  = "GREEDYNMM"
POSTPROCESS_METRIC = "IOU"
POSTPROCESS_THRESH = 0.50
MIN_BOX_AREA      = 400

DISPLAY_SCALE = 0.6
HUD_LINE_H    = 28
HUD_FONT      = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE     = 0.6
HUD_THICK     = 1


# LOAD YOLO
model = YOLO("runs/detect/train9/weights/best.pt").to(DEVICE)
model.to(DEVICE)
model_path = model.ckpt_path

sahi_model = None

def get_sahi_model():
    global sahi_model
    if sahi_model is None:
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=MODEL_CONFIDENCE_FLOOR,
            device=DEVICE,
        )
    return sahi_model

# VIDEO
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {video_path}")

vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
delay = max(1, int(1000.0 / fps))

cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Detection", int(vid_w * DISPLAY_SCALE), int(vid_h * DISPLAY_SCALE))

# HUD
def draw_hud(frame, frame_idx, kept, total_cands):
    lines = [
        "=== LIVE DETECTION ===",
        "",
        f"[S] SAM: {'ON' if USE_SAM else 'OFF'}",
        f"[M] Mode: {'SAHI' if USE_SAHI else 'YOLO'}",
        f"[T] Temporal: {'ON' if TEMPORAL_ENABLED else 'OFF'}",
        "",
        f"Detections: {kept} (raw {total_cands})",
        f"Frame: {frame_idx}/{total}",
        "",
        "[1/2/3] Confidence",
        "[Q] Quit"
    ]

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

# MAIN LOOP
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1


    def box_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


    def should_update_sam(entries, last_boxes, frame_idx):
        if frame_idx % SAM_EVERY_N_FRAMES == 0:
            return True

        if len(entries) != len(last_boxes):
            return True

        for pred, old_box in zip(entries, last_boxes):
            cx1, cy1 = box_center(pred["bbox"])
            cx2, cy2 = box_center(old_box)

            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

            if dist > SAM_MOVE_THRESHOLD:
                return True

        return False
    # DETECTION
    if USE_SAHI:
        result = get_sliced_prediction(frame, get_sahi_model(),
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_H,
            overlap_width_ratio=OVERLAP_W)

        raw = [
            {"bbox": [p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy],
             "class_id": p.category.id,
             "score": p.score.value}
            for p in result.object_prediction_list
        ]
    else:
        res = model.predict(
            frame,
            conf=MODEL_CONFIDENCE_FLOOR,
            device=0,
            imgsz=640,
            half=True,
            verbose=False
        )[0]

        raw = []
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw.append({
                    "bbox": [x1,y1,x2,y2],
                    "class_id": int(box.cls),
                    "score": float(box.conf)
                })

    # FILTER
    entries = []
    for pred in raw:
        if pred["score"] < CONFIDENCE_THRESHOLD:
            continue
        x1,y1,x2,y2 = pred["bbox"]
        if (x2-x1)*(y2-y1) < MIN_BOX_AREA:
            continue
        entries.append(pred)

    # SAM PART
    if USE_SAM and len(entries) > 0:
        top_entries = sorted(entries, key=lambda x: -x["score"])[:2]

        if should_update_sam(top_entries, last_sam_boxes, frame_idx):
            predictor = get_sam_predictor()
            predictor.set_image(frame)

            cached_sam_masks = []
            last_sam_boxes = []

            for pred in top_entries:
                x1, y1, x2, y2 = map(int, pred["bbox"])
                input_box = np.array([x1, y1, x2, y2])

                masks, _, _ = predictor.predict(
                    box=input_box,
                    multimask_output=False
                )

                cached_sam_masks.append(masks[0])
                last_sam_boxes.append(pred["bbox"])

        for mask in cached_sam_masks:
            frame[mask] = frame[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

    # ----------------------------
    # DRAW BOXES
    # ----------------------------
    for pred in entries:
        x1,y1,x2,y2 = map(int, pred["bbox"])
        name = model.names.get(pred["class_id"], str(pred["class_id"]))
        label = f"{name} {pred['score']:.2f}"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,label,(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    draw_hud(frame, frame_idx, len(entries), len(raw))

    cv2.imshow("Live Detection", frame)

    # KEYS
    key = cv2.waitKey(delay) & 0xFF

    if key in (ord("q"), 27):
        break
    elif key == ord("s"):
        USE_SAM = not USE_SAM
        print("SAM:", USE_SAM)
    elif key == ord("m"):
        USE_SAHI = not USE_SAHI
    elif key == ord("t"):
        TEMPORAL_ENABLED = not TEMPORAL_ENABLED
    elif key == ord("1"):
        CONFIDENCE_THRESHOLD = CONFIDENCE_PRESETS[0]
    elif key == ord("2"):
        CONFIDENCE_THRESHOLD = CONFIDENCE_PRESETS[1]
    elif key == ord("3"):
        CONFIDENCE_THRESHOLD = CONFIDENCE_PRESETS[2]

cap.release()
cv2.destroyAllWindows()