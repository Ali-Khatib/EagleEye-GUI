from multiprocessing import freeze_support

import torch
from ultralytics import YOLO


def main():
    device = 0
    print(f"[INFO] Using device: cuda:{device}")
    print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}")

    model = YOLO("yolo11n.pt")

    model.train(
        data="kitti.yaml",
        epochs=50,
        imgsz=640,
        batch=32,
        device=device,
        workers=2,
        cache=False,
        amp=True,
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
    print("[DONE] Training complete")
    print("Best model: runs/detect/train/weights/best.pt")
