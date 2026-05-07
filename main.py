"""Image demo: SAHI detections + SAM segmentation."""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO, SAM
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    return image


def get_sahi_model(model_path: str, device: str, conf_floor: float) -> AutoDetectionModel:
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf_floor,
        device=device,
    )


def sahi_detect(
    image: np.ndarray,
    sahi_model: AutoDetectionModel,
    slice_height: int,
    slice_width: int,
    overlap_h: float,
    overlap_w: float,
    postprocess_type: str,
    postprocess_metric: str,
    postprocess_thresh: float,
    conf: float,
) -> list[dict]:
    result = get_sliced_prediction(
        image,
        sahi_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_h,
        overlap_width_ratio=overlap_w,
        postprocess_type=postprocess_type,
        postprocess_match_metric=postprocess_metric,
        postprocess_match_threshold=postprocess_thresh,
    )
    out: list[dict] = []
    for pred in result.object_prediction_list:
        score = float(pred.score.value)
        if score < conf:
            continue
        out.append({
            "bbox": [float(pred.bbox.minx), float(pred.bbox.miny), float(pred.bbox.maxx), float(pred.bbox.maxy)],
            "class_id": int(pred.category.id),
            "score": score,
        })
    return out


def segment_with_sam(sam: SAM, image: np.ndarray, bboxes: np.ndarray):
    if bboxes.size == 0:
        return None
    return sam(image, bboxes=bboxes.tolist(), verbose=False)


def draw_overlay(
    image: np.ndarray,
    detections: list[dict],
    masks,
    class_names: dict,
) -> np.ndarray:
    overlay = image.copy()
    num_det = len(detections)
    if num_det == 0:
        return overlay

    np.random.seed(42)
    colors = np.random.randint(60, 255, size=(num_det, 3)).tolist()

    if masks and masks[0].masks is not None:
        mask_data = masks[0].masks.data.cpu().numpy()
        for i, mask in enumerate(mask_data):
            color = colors[i % len(colors)]
            colored_mask = np.zeros_like(overlay, dtype=np.uint8)
            colored_mask[mask > 0.5] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.45, 0)

            binary = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = colors[i % len(colors)]
        name = class_names.get(det["class_id"], str(det["class_id"]))
        label = f"{name} {det['score']:.2f}"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            overlay,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return overlay


def run_pipeline(
    image_path: str,
    yolo_model_path: str,
    sam_model_path: str,
    conf: float,
    device: str,
    slice_height: int,
    slice_width: int,
    overlap_h: float,
    overlap_w: float,
    postprocess_type: str,
    postprocess_metric: str,
    postprocess_thresh: float,
    output_path: str | None,
    no_display: bool,
):
    print(f"Device      : {device}")
    print(f"Image       : {image_path}")
    print(f"YOLO model  : {yolo_model_path}")
    print(f"SAM model   : {sam_model_path}")
    print("Mode        : SAHI + SAM")

    image = load_image(image_path)
    yolo = YOLO(yolo_model_path)
    yolo.to(device)
    sam = SAM(sam_model_path)

    print("Running SAHI detection...")
    sahi_model = get_sahi_model(yolo_model_path, device, conf_floor=0.01)
    detections = sahi_detect(
        image,
        sahi_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_h=overlap_h,
        overlap_w=overlap_w,
        postprocess_type=postprocess_type,
        postprocess_metric=postprocess_metric,
        postprocess_thresh=postprocess_thresh,
        conf=conf,
    )

    if not detections:
        print("No detections. Nothing to segment.")
        return

    bboxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    print(f"Detections  : {len(detections)}")
    print("Running SAM segmentation...")
    sam_results = segment_with_sam(sam, image, bboxes)

    overlay = draw_overlay(image, detections, sam_results, yolo.names)

    if output_path:
        cv2.imwrite(output_path, overlay)
        print(f"Saved       : {output_path}")

    if not no_display:
        h, w = overlay.shape[:2]
        scale = min(1.0, 1400.0 / w, 900.0 / h)
        if scale < 1.0:
            overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)))
        cv2.imshow("SAHI + SAM", overlay)
        print("Press any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAHI detections + SAM segmentation")
    parser.add_argument("--image", default="demo_data/small-vehicles1.jpeg")
    parser.add_argument("--yolo-model", default="runs/detect/train8/weights/best.pt")
    parser.add_argument("--sam-model", default="sam2.1_l.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--slice-height", type=int, default=512)
    parser.add_argument("--slice-width", type=int, default=512)
    parser.add_argument("--overlap-h", type=float, default=0.2)
    parser.add_argument("--overlap-w", type=float, default=0.2)
    parser.add_argument("--small-objects", action="store_true", default=True)
    parser.add_argument("--no-small-objects", dest="small_objects", action="store_false")
    parser.add_argument("--postprocess-type", default="GREEDYNMM")
    parser.add_argument("--postprocess-metric", default="IOU")
    parser.add_argument("--postprocess-thresh", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-display", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.small_objects:
        # SAHI small-object preset: smaller tiles + more overlap
        args.slice_height = 384
        args.slice_width = 384
        args.overlap_h = 0.25
        args.overlap_w = 0.25
    output_path = args.output or "demo_data/sahi_sam_result.png"
    run_pipeline(
        image_path=args.image,
        yolo_model_path=args.yolo_model,
        sam_model_path=args.sam_model,
        conf=args.conf,
        device=args.device,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_h=args.overlap_h,
        overlap_w=args.overlap_w,
        postprocess_type=args.postprocess_type,
        postprocess_metric=args.postprocess_metric,
        postprocess_thresh=args.postprocess_thresh,
        output_path=output_path,
        no_display=args.no_display,
    )


if __name__ == "__main__":
    main()
