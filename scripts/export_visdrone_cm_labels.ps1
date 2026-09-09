# Export YOLO prediction txt files for confusion-matrix build (one-time GPU pass).
# Does NOT recompute metrics or overwrite metrics.json — only writes <mode>/labels/*.txt
#
# After this finishes, build PNGs instantly:
#   python scripts/build_visdrone_confusion_matrices.py

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$src = "dataset/VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/images"
$gt  = "dataset/VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/labels"
$yolo11 = "runs/detect/train9/weights/best.pt"
$yolov8 = "runs/detect/train_v8_visdrone/weights/best.pt"

Write-Host "[1/2] YOLO11 pipelines (modes 1-5) -> visdrone_test_dev/labels/"
python pipelines/compare_all.py `
    --source $src `
    --ground-truth $gt `
    --weights $yolo11 `
    --output "benchmark_working/supervisor_report/visdrone_test_dev" `
    --modes "yolo_only,yolo_sahi,sam3_only,sam3_yolo,yolo_sahi_sam3" `
    --labels-only `
    --resume `
    --skip-images

Write-Host "[2/2] YOLOv8 yolo_sahi_sam3 (mode 6) -> visdrone_test_dev_yolov8/labels/"
python pipelines/compare_all.py `
    --source $src `
    --ground-truth $gt `
    --weights $yolov8 `
    --output "benchmark_working/supervisor_report/visdrone_test_dev_yolov8" `
    --modes "yolo_sahi_sam3" `
    --labels-only `
    --resume `
    --skip-images

Write-Host ""
Write-Host "[DONE] Labels exported. Build confusion matrices (no GPU):"
Write-Host "  python scripts/build_visdrone_confusion_matrices.py"
