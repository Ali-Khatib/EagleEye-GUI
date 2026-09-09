# Train YOLOv8n on VisDrone then benchmark all 5 pipelines on test-dev.
# Log: runs/detect/train_v8_visdrone/train.log (ultralytics) + benchmark_working/.../benchmark.log

param(
    [int]$Epochs = 100,
    [int]$Limit = 0,
    [switch]$SkipTrain,
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
$weights = Join-Path $root "runs\detect\train_v8_visdrone\weights\best.pt"
$log = Join-Path $root "benchmark_working\visdrone_yolov8_pipeline.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Host $line
}

Set-Location $root

if (-not $SkipTrain -and -not (Test-Path $weights)) {
    Log "Training YOLOv8n on VisDrone ($Epochs epochs)..."
    & $py -u (Join-Path $root "scripts\training\train_yolo_visdrone_v8.py") --epochs $Epochs 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "Training failed" }
    Log "Training done: $weights"
} elseif (Test-Path $weights) {
    Log "Using existing weights: $weights"
} else {
    throw "Weights missing and -SkipTrain set"
}

$benchArgs = @()
if ($Limit -gt 0) { $benchArgs += "-Limit", $Limit }
if ($SkipExisting) { $benchArgs += "-SkipExisting" }

Log "Starting VisDrone test-dev benchmark (YOLOv8)..."
& (Join-Path $root "scripts\run_visdrone_yolov8_benchmark.ps1") @benchArgs 2>&1 | Tee-Object -FilePath $log -Append
Log "Complete. See comparison_report\VisDrone YOLOv8 report.txt"
