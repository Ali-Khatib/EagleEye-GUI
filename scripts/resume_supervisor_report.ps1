# Resume the professor/supervisor benchmark from the last checkpoint.
# Shows tqdm progress bars in the terminal (image-level + dataset-level).
#
# Current state (benchmark_working/supervisor_report/kitti_val):
#   DONE: yolo_only, yolo_sahi, sam3_only, sam3_yolo  (1496 images each)
#   TODO: yolo_sahi_sam3 on KITTI, then all 5 on VisDrone test-dev
#
# Usage (from repo root):
#   .\scripts\resume_supervisor_report.ps1
#   .\scripts\resume_supervisor_report.ps1 -Limit 10   # smoke test

param(
    [int]$Limit = 0,
    [string]$Modes = "sam3_yolo,yolo_sahi_sam3",
    [switch]$SkipVisDrone,
    [switch]$VisDroneOnly
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
$out = Join-Path $root "benchmark_working\supervisor_report"
$log = Join-Path $out "resume_benchmark.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Host $line
}

function Invoke-PythonLogged {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PyArgs)
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $py -u @PyArgs 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Python exited with code $LASTEXITCODE"
        }
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

Set-Location $root
New-Item -ItemType Directory -Force -Path $out | Out-Null

Log "=== RESUME supervisor benchmark ==="
Log "Output: $out"

$limitArgs = @()
if ($Limit -gt 0) { $limitArgs = @("--limit", "$Limit") }

if (-not $VisDroneOnly) {
Log "Step 1/2: KITTI val modes ($Modes)"
Invoke-PythonLogged `
    (Join-Path $root "pipelines\compare_all.py") `
    --source (Join-Path $root "kitti\images\val") `
    --ground-truth (Join-Path $root "kitti\labels\val") `
    --weights (Join-Path $root "runs\detect\train2\weights\best.pt") `
    --output (Join-Path $out "kitti_val") `
    --device cuda `
    --skip-images `
    --skip-existing `
    --dataset-name "KITTI validation (labeled eval split)" `
    --modes $Modes `
    @limitArgs
} else {
Log "Skipping KITTI (VisDroneOnly set)"
}

if (-not $SkipVisDrone) {
Log "Step 2/2: VisDrone test-dev all 5 pipelines"
Invoke-PythonLogged `
    (Join-Path $root "scripts\run_supervisor_report.py") `
    --device cuda `
    --output-root $out `
    --datasets visdrone_test_dev `
    --skip-existing `
    @limitArgs

Log "Regenerating comparison_report/"
& (Join-Path $root "scripts\build_comparison_report.ps1")
} else {
Log "Skipping VisDrone (SkipVisDrone set)"
}

Log "DONE reports at:"
Log "  $out\SUPERVISOR_REPORT.csv"
Log "  $out\SUPERVISOR_REPORT.md"
