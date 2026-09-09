import os
import shutil
import random
from PIL import Image

# KITTI folders
IMAGE_DIR = r"C:\Users\khati\PycharmProjects\SAHI\kitti\data_object_image_2\training\image_2"
LABEL_DIR = r"C:\Users\khati\PycharmProjects\SAHI\kitti\data_object_label_2\training\label_2"

OUT_IMAGES_TRAIN = r"C:\Users\khati\PycharmProjects\SAHI\kitti\images\train"
OUT_IMAGES_VAL = r"C:\Users\khati\PycharmProjects\SAHI\kitti\images\val"
OUT_LABELS_TRAIN = r"C:\Users\khati\PycharmProjects\SAHI\kitti\labels\train"
OUT_LABELS_VAL = r"C:\Users\khati\PycharmProjects\SAHI\kitti\labels\val"

CLASSES = ["Car", "Pedestrian", "Cyclist"]

VAL_SPLIT = 0.2

for folder in [OUT_IMAGES_TRAIN, OUT_IMAGES_VAL, OUT_LABELS_TRAIN, OUT_LABELS_VAL]:
    os.makedirs(folder, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
random.shuffle(image_files)

val_count = int(len(image_files) * VAL_SPLIT)
val_files = set(image_files[:val_count])

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_file = img_file.replace(".png", ".txt")
    kitti_label_path = os.path.join(LABEL_DIR, label_file)

    if img_file in val_files:
        out_img_dir = OUT_IMAGES_VAL
        out_label_dir = OUT_LABELS_VAL
    else:
        out_img_dir = OUT_IMAGES_TRAIN
        out_label_dir = OUT_LABELS_TRAIN

    shutil.copy(img_path, os.path.join(out_img_dir, img_file))

    img = Image.open(img_path)
    w, h = img.size

    yolo_lines = []

    if os.path.exists(kitti_label_path):
        with open(kitti_label_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 8:
                    continue

                cls = parts[0]

                if cls not in CLASSES:
                    continue

                cls_id = CLASSES.index(cls)

                xmin = float(parts[4])
                ymin = float(parts[5])
                xmax = float(parts[6])
                ymax = float(parts[7])

                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                box_width = (xmax - xmin) / w
                box_height = (ymax - ymin) / h

                yolo_lines.append(
                    f"{cls_id} {x_center} {y_center} {box_width} {box_height}"
                )

    with open(os.path.join(out_label_dir, label_file), "w") as out:
        out.write("\n".join(yolo_lines))

print("Done. Dataset converted to YOLO format.")