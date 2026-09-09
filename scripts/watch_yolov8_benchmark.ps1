# Live tqdm-style view of YOLOv8 VisDrone benchmark progress.
# Run in a separate terminal while training/benchmark is running in background.
#
#   .\scripts\watch_yolov8_benchmark.ps1

$root = Split-Path $PSScriptRoot -Parent
$progress = Join-Path $root "benchmark_working\yolov8_benchmark_progress.txt"
$live = Join-Path $root "benchmark_working\yolov8_benchmark_live.log"

Write-Host "Watching YOLOv8 benchmark progress"
Write-Host "  progress bar: $progress"
Write-Host "  full log:     $live"
Write-Host "Press Ctrl+C to stop watching (benchmark keeps running)"
Write-Host ""

$last = ""
while ($true) {
    if (Test-Path $progress) {
        $line = (Get-Content $progress -ErrorAction SilentlyContinue | Select-Object -Last 1)
        if ($line -and $line -ne $last) {
            $last = $line
            Write-Host "`r$line" -NoNewline
        }
    } elseif (Test-Path $live) {
        $tail = Get-Content $live -Tail 1 -ErrorAction SilentlyContinue
        if ($tail -and $tail -match "Benchmark|yolo_sahi_sam3|%\|") {
            Write-Host "`r$tail" -NoNewline
        }
    } else {
        Write-Host "`r[WAIT] Benchmark not started yet..." -NoNewline
    }
    Start-Sleep -Milliseconds 800
}
