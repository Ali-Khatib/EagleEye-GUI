import cv2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from collections import deque


video_path = "demo_data/test.mp4"
output_path = "demo_data/test2_out.mp4"
SAVE_OUTPUT_VIDEO = True


CONFIDENCE_PRESETS      = [0.35, 0.10, 0.05]
PRESET_LABELS           = ["HIGH (0.20)", "MID  (0.10)", "LOW  (0.05)"]
confidence_preset_index = 1
CONFIDENCE_THRESHOLD    = CONFIDENCE_PRESETS[confidence_preset_index]
MODEL_CONFIDENCE_FLOOR  = 0.01

USE_SAHI             = False
TEMPORAL_ENABLED     = False
TEMPORAL_WINDOW      = 3
TEMPORAL_MIN_HITS    = 2
TEMPORAL_DIST_RATIO  = 0.75
recent_history       = deque(maxlen=TEMPORAL_WINDOW - 1)

SLICE_HEIGHT      = 512
SLICE_WIDTH       = 512
OVERLAP_H         = 0.25
OVERLAP_W         = 0.25
POSTPROCESS_TYPE  = "GREEDYNMM"
POSTPROCESS_METRIC = "IOU"
POSTPROCESS_THRESH = 0.90

DISPLAY_SCALE = 0.6
HUD_LINE_H    = 28
HUD_FONT      = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE     = 0.6
HUD_THICK     = 1

# Load model
model = YOLO("yolo11n.pt")
model_path = model.ckpt_path
print(f"Model: {model_path}")

sahi_model = None


def get_sahi_model():
    global sahi_model
    if sahi_model is None:
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=MODEL_CONFIDENCE_FLOOR,
            device="cpu",
        )
    return sahi_model

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {video_path}")

vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
delay = max(1, int(1000.0 / fps))

print(f"Video: {vid_w}x{vid_h} @ {fps:.1f}fps  ({total} frames)")

out = None
if SAVE_OUTPUT_VIDEO:
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (vid_w, vid_h),
    )

cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Detection", int(vid_w * DISPLAY_SCALE), int(vid_h * DISPLAY_SCALE))


# HUD drawing
def draw_hud(frame, frame_idx, kept, total_cands):
    lines = [
        ("=== LIVE DETECTION ===",  None),
        ("",                        None),
        ("CONFIDENCE:",             None),
    ]
    for i, label in enumerate(PRESET_LABELS):
        active = (i == confidence_preset_index)
        prefix = " >>>" if active else "    "
        lines.append((f"{prefix} [{i+1}] {label}", active))
    lines += [
        ("", None),
        ("TOGGLES:", None),
        (f"  [T] Temporal : {'ON ' if TEMPORAL_ENABLED else 'OFF'}", TEMPORAL_ENABLED),
        (f"  [M] Mode     : {'SAHI sliced  ' if USE_SAHI else 'YOLO realtime'}", USE_SAHI),
        ("", None),
        ("  [Q] Quit", None),
        ("", None),
        (f"  Detections : {kept}  (raw: {total_cands})", None),
        (f"  Frame      : {frame_idx}/{total}", None),
    ]

    pad     = 12
    panel_w = 340
    panel_h = len(lines) * HUD_LINE_H + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (text, state) in enumerate(lines):
        if not text:
            continue
        if state is None:
            color = (0, 220, 255)
        elif state:
            color = (0, 255, 100)
        else:
            color = (200, 200, 200)
        y = pad + (i + 1) * HUD_LINE_H
        cv2.putText(frame, text, (pad + 8, y), HUD_FONT, HUD_SCALE, color, HUD_THICK, cv2.LINE_AA)


# Main loop

frame_idx = 0
print("Running — Q/ESC=quit  1/2/3=confidence  T=temporal  M=mode")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_idx += 1

    # Inference
    if USE_SAHI:
        result = get_sliced_prediction(
            frame, get_sahi_model(),
            slice_height=SLICE_HEIGHT, slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_H, overlap_width_ratio=OVERLAP_W,
            postprocess_type=POSTPROCESS_TYPE,
            postprocess_match_metric=POSTPROCESS_METRIC,
            postprocess_match_threshold=POSTPROCESS_THRESH,
        )
        raw = [
            {"bbox": [float(p.bbox.minx), float(p.bbox.miny),
                      float(p.bbox.maxx), float(p.bbox.maxy)],
             "class_id": int(p.category.id), "score": float(p.score.value)}
            for p in result.object_prediction_list
        ]
    else:
        res = model.predict(frame, conf=MODEL_CONFIDENCE_FLOOR, verbose=False)[0]
        raw = []
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw.append({"bbox": [x1, y1, x2, y2],
                             "class_id": int(box.cls.item()),
                             "score": float(box.conf.item())})

    total_cands = len(raw)

    # Confidence filter
    entries = []
    for pred in raw:
        if pred["score"] < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = pred["bbox"]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        entries.append({
            "pred": pred,
            "class_id": pred["class_id"],
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "scale": max(bw, bh, 1.0),
        })

    # Temporal filter
    if TEMPORAL_ENABLED and recent_history:
        confirmed = []
        for e in entries:
            hits = 1
            for prev_frame in recent_history:
                for prev in prev_frame:
                    if prev["class_id"] != e["class_id"]:
                        continue
                    max_d = TEMPORAL_DIST_RATIO * max(e["scale"], prev["scale"], 1.0)
                    dx = e["cx"] - prev["cx"]
                    dy = e["cy"] - prev["cy"]
                    if dx * dx + dy * dy <= max_d * max_d:
                        hits += 1
                        break
            if hits >= TEMPORAL_MIN_HITS:
                confirmed.append(e)
        final_entries = confirmed if confirmed else entries
    else:
        final_entries = entries

    recent_history.append([
        {"class_id": e["class_id"], "cx": e["cx"],
         "cy": e["cy"], "scale": e["scale"]}
        for e in entries
    ])

    # Draw boxes
    for e in final_entries:
        x1, y1, x2, y2 = e["pred"]["bbox"]
        name  = model.names.get(e["class_id"], str(e["class_id"]))
        label = f"{name} {e['pred']['score']:.2f}"
        pt1   = (int(x1), int(y1))
        pt2   = (int(x2), int(y2))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(frame, label, (pt1[0], max(18, pt1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    draw_hud(frame, frame_idx, len(final_entries), total_cands)

    if out is not None:
        out.write(frame)

    cv2.imshow("Live Detection", frame)

    # Key handling — instant response
    key = cv2.waitKey(delay) & 0xFF
    if key in (ord("q"), 27):
        print("Stopped by user.")
        break
    elif key == ord("1"):
        confidence_preset_index = 0
        CONFIDENCE_THRESHOLD = CONFIDENCE_PRESETS[0]
        print(f"Confidence → {CONFIDENCE_THRESHOLD}")
    elif key == ord("2"):
        confidence_preset_index = 1
        CONFIDENCE_THRESHOLD = CONFIDENCE_PRESETS[1]
        print(f"Confidence → {CONFIDENCE_THRESHOLD}")
    elif key == ord("3"):
        confidence_preset_index = 2
        CONFIDENCE_THRESHOLD = CONFIDENCE_PRESETS[2]
        print(f"Confidence → {CONFIDENCE_THRESHOLD}")
    elif key in (ord("t"), ord("T")):
        TEMPORAL_ENABLED = not TEMPORAL_ENABLED
        print(f"Temporal → {'ON' if TEMPORAL_ENABLED else 'OFF'}")
    elif key in (ord("m"), ord("M")):
        USE_SAHI = not USE_SAHI
        print(f"Mode → {'SAHI sliced' if USE_SAHI else 'YOLO realtime'}")

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
print(f"Done. Saved: {output_path}")
