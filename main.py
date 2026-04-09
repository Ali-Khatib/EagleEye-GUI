from sahi.utils.file import download_from_url
from ultralytics import YOLO

# Load latest YOLO model (auto-downloads if not present)
model = YOLO("yolo11n.pt")  # nano model (fast)

model_path = model.ckpt_path
print("Model downloaded at:", model_path)

# Download test images
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
    "demo_data/terrain2.png",
)

from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.3,
    device="cpu",
)

from sahi.predict import get_prediction
from sahi.utils.cv import read_image

# With an image path
result = get_prediction("demo_data/small-vehicles1.jpeg", detection_model)

# With a numpy image
result_with_np_image = get_prediction(read_image("demo_data/small-vehicles1.jpeg"), detection_model)



from IPython.display import Image

result.export_visuals(export_dir="demo_data/")
Image("demo_data/prediction_visual.png")



from sahi.predict import get_sliced_prediction

result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height=128,
    slice_width=128,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

object_prediction_list = result.object_prediction_list

result.to_coco_annotations()[:3]
result.to_coco_predictions(image_id=1)[:3]
result.to_imantics_annotations()[:3]
result.to_fiftyone_detections()[:3]


from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path=model_path,
    model_device="cpu",
    model_confidence_threshold=0.4,
    source="demo_data/small-vehicles1.jpeg",
    slice_height=128,
    slice_width=128,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

