# VisDrone test-dev benchmark with YOLOv8n (same 5 pipelines, for paper comparison vs SAHI+YOLOv8).
#
# Prerequisite: train YOLOv8 first
#   .\.venv\Scripts\python.exe scripts\training\train_yolo_visdrone_v8.py
#
# Usage:
#   .\scripts\run_visdrone_yolov8_benchmark.ps1
#   .\scripts\run_visdrone_yolov8_benchmark.ps1 -Limit 10
#   .\scripts\run_visdrone_yolov8_benchmark.ps1 -Modes "yolo_sahi,yolo_sahi_sam3"

param(
    [int]$Limit = 0,
    [string]$Weights = "",
    [string]$Modes = "yolo_sahi_sam3",
    [switch]$SkipExisting,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"

# Confirm NVIDIA GPU is visible to PyTorch before a long benchmark run.
$cudaCheck = & $py -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')" 2>&1
if ($cudaCheck[-1] -eq "NO_GPU" -or $cudaCheck[0] -ne "True") {
    Write-Host "[ERROR] CUDA GPU not available to PyTorch. Benchmark needs your RTX 4060." -ForegroundColor Red
    Write-Host $cudaCheck
    exit 1
}
Write-Host "GPU: $($cudaCheck[-1])"

if (-not $Weights) {
    $Weights = Join-Path $root "runs\detect\train_v8_visdrone\weights\best.pt"
}
if (-not (Test-Path $Weights)) {
    Write-Host "[ERROR] YOLOv8 weights not found: $Weights" -ForegroundColor Red
    Write-Host "Train first: $py scripts\training\train_yolo_visdrone_v8.py"
    exit 1
}

$src = Join-Path $root "dataset\VisDrone2019-DET-test-dev\VisDrone2019-DET-test-dev\images"
$gt = Join-Path $root "dataset\VisDrone2019-DET-test-dev\VisDrone2019-DET-test-dev\labels"
$out = Join-Path $root "benchmark_working\supervisor_report\visdrone_test_dev_yolov8"
$progressLog = Join-Path $root "benchmark_working\yolov8_benchmark_progress.txt"
$liveLog = Join-Path $root "benchmark_working\yolov8_benchmark_live.log"

$labels = Get-ChildItem $gt -Filter "*.txt" -ErrorAction SilentlyContinue
if (-not $labels) {
    Write-Host "[INFO] Building VisDrone test-dev labels..."
    & $py (Join-Path $root "scripts\training\visdrone_to_yolo.py") --split test-dev
}

$args = @(
    (Join-Path $root "pipelines\compare_all.py"),
    "--source", $src,
    "--ground-truth", $gt,
    "--weights", $Weights,
    "--output", $out,
    "--device", "cuda",
    "--skip-images",
    "--dataset-name", "VisDrone test-dev (YOLOv8n)",
    "--modes", $Modes,
    "--progress-log", $progressLog
)
if ($Limit -gt 0) { $args += @("--limit", "$Limit") }
if ($SkipExisting) { $args += "--skip-existing" }
if ($Resume) { $args += "--resume" }

Write-Host "YOLOv8 weights: $Weights"
Write-Host "Output: $out"
Write-Host "Modes: $Modes"
Write-Host ""
Write-Host "Live progress bar (open another terminal):"
Write-Host "  .\scripts\watch_yolov8_benchmark.ps1"
Write-Host ""
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $py -u @args 2>&1 | Tee-Object -FilePath $liveLog | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "compare_all.py exited with code $LASTEXITCODE"
    }
} finally {
    $ErrorActionPreference = $prevEap
}

& $py (Join-Path $root "scripts\build_yolov8_comparison_report.py")
Write-Host "[DONE] Report: comparison_report\VisDrone YOLOv8 report.txt"
