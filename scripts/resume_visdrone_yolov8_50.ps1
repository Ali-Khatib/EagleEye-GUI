# Save current YOLOv8 checkpoints and resume training to 50 epochs total (not 100).
param(
    [int]$Epochs = 50,
    [string]$RunName = "train_v8_visdrone"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
$runDir = Join-Path $root "runs\detect\$RunName"
$weights = Join-Path $runDir "weights"
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$backup = Join-Path $root "runs\detect\${RunName}_backup_$stamp"
$statusLog = Join-Path $root "benchmark_working\yolov8_resume.log"

function Log($m) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m"
    Write-Host $line
    New-Item -ItemType Directory -Force -Path (Split-Path $statusLog) | Out-Null
    Add-Content $statusLog $line -ErrorAction SilentlyContinue
}

Log "=== Switching YOLOv8 training to $Epochs epochs (resume) ==="

# Stop active training / waiter shells first (releases yolov8_train.log lock)
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -match "train_yolo_visdrone_v8" -or
        $_.CommandLine -match "wait_and_run_yolov8_benchmark" -or
        $_.CommandLine -match "resume_visdrone_yolov8_50"
    } |
    Where-Object { $_.ProcessId -ne $PID } |
    ForEach-Object {
        Log "Stopping powershell PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "train_yolo_visdrone_v8" } |
    ForEach-Object {
        Log "Stopping python PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Start-Sleep -Seconds 8

if (Test-Path $weights) {
    New-Item -ItemType Directory -Force -Path $backup | Out-Null
    Copy-Item (Join-Path $weights "best.pt") $backup -Force -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $weights "last.pt") $backup -Force -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $runDir "args.yaml") $backup -Force -ErrorAction SilentlyContinue
    Log "Saved checkpoint backup: $backup"
}

Log "Resuming from last.pt to epoch $Epochs ..."
Set-Location $root
& $py -u (Join-Path $root "scripts\training\train_yolo_visdrone_v8.py") `
    --epochs $Epochs --resume --name $RunName 2>&1 |
    Tee-Object -FilePath (Join-Path $root "benchmark_working\yolov8_train_resume.log") -Append

if ($LASTEXITCODE -ne 0) { throw "Resume training failed with code $LASTEXITCODE" }
Log "[DONE] YOLOv8 training finished at $Epochs epochs."
