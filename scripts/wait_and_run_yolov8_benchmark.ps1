# Wait for YOLOv8 training to finish, then run VisDrone test-dev yolo_sahi_sam3 benchmark.
param(
    [int]$Epochs = 50
)

$root = Split-Path $PSScriptRoot -Parent
$weights = Join-Path $root "runs\detect\train_v8_visdrone\weights\best.pt"
$trainLog = Join-Path $root "benchmark_working\yolov8_train_resume.log"
$trainLogAlt = Join-Path $root "benchmark_working\yolov8_train.log"
$log = Join-Path $root "benchmark_working\yolov8_benchmark_wait.log"

function Log($m) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m"
    New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
    Add-Content $log $line
    Write-Host $line
}

function Training-CompleteInLog {
    $text = ""
    if (Test-Path $trainLog) {
        $text += Get-Content $trainLog -Raw -ErrorAction SilentlyContinue
    }
    if (Test-Path $trainLogAlt) {
        $text += Get-Content $trainLogAlt -Raw -ErrorAction SilentlyContinue
    }
    if (-not $text) { return $false }
    if ($text -match "\[DONE\] Best weights") { return $true }
    if ($text -match "\[DONE\] YOLOv8 training finished") { return $true }
    if ($text -match "${Epochs}/${Epochs}\s+\S+\s+[\d.]+G") { return $true }
    if ($text -match "Stopped at epoch $Epochs") { return $true }
    return $false
}

Log "Waiting for YOLOv8 training to finish ($Epochs epochs)..."
while (-not (Training-CompleteInLog)) {
    if (Test-Path $trainLog) {
        $tail = (Get-Content $trainLog -Tail 3 -ErrorAction SilentlyContinue) -join " "
        if ($tail -match "(\d+)/$Epochs") { Log "Training progress: epoch $($Matches[1])/$Epochs" }
        elseif ($tail -match "(\d+)/100") { Log "Training progress: epoch $($Matches[1])/100 (switching to $Epochs target)" }
    }
    Start-Sleep -Seconds 120
}

Start-Sleep -Seconds 15
Log "Starting VisDrone test-dev benchmark (YOLOv8, yolo_sahi_sam3 only)."
Log "Open another terminal and run: .\scripts\watch_yolov8_benchmark.ps1"
Set-Location $root
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & (Join-Path $root "scripts\run_visdrone_yolov8_benchmark.ps1") -Modes "yolo_sahi_sam3" -SkipExisting 2>&1 |
        Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark script exited with code $LASTEXITCODE"
    }
} finally {
    $ErrorActionPreference = $prevEap
}
Log "Benchmark complete."
