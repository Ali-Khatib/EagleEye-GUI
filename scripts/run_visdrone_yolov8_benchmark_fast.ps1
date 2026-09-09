# Fast YOLOv8 + SAHI + SAM3 on VisDrone test-dev (1610 images).
#
# Speedups vs default run:
#   - SAHI model cached once (not reloaded per image)
#   - Per-image checkpoint (safe to kill / resume with -Resume)
#   - Larger SAHI tiles (640px, 10% overlap) — fewer slices per image
#   - skip-images (no disk writes during eval)
#
# Usage:
#   .\scripts\run_visdrone_yolov8_benchmark_fast.ps1
#   .\scripts\run_visdrone_yolov8_benchmark_fast.ps1 -Resume
#   .\scripts\watch_yolov8_benchmark.ps1

param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:SAHI_SLICE = "640"
$env:SAHI_OVERLAP = "0.10"

$runArgs = @{
    Modes        = "yolo_sahi_sam3"
    SkipExisting = $true
}
if ($Resume) {
    $runArgs.Resume = $true
}

Write-Host "Fast YOLOv8 benchmark: yolo_sahi_sam3 only"
Write-Host "  SAHI_SLICE=$env:SAHI_SLICE  SAHI_OVERLAP=$env:SAHI_OVERLAP"
Write-Host "  checkpoint: benchmark_working\supervisor_report\visdrone_test_dev_yolov8\yolo_sahi_sam3\benchmark_checkpoint.jsonl"
if ($Resume) {
    Write-Host "  mode: RESUME from checkpoint"
} else {
    Write-Host "  mode: fresh run (use -Resume if interrupted)"
}
Write-Host ""

& (Join-Path $PSScriptRoot "run_visdrone_yolov8_benchmark.ps1") @runArgs
