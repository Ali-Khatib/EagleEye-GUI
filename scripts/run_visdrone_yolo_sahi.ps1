$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "benchmark_working\supervisor_report\yolo_sahi_rerun.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $py -u (Join-Path $root "pipelines\compare_all.py") `
        --source (Join-Path $root "dataset\VisDrone2019-DET-test-dev\VisDrone2019-DET-test-dev\images") `
        --ground-truth (Join-Path $root "dataset\VisDrone2019-DET-test-dev\VisDrone2019-DET-test-dev\labels") `
        --weights (Join-Path $root "runs\detect\train9\weights\best.pt") `
        --output (Join-Path $root "benchmark_working\supervisor_report\visdrone_test_dev") `
        --device cuda `
        --skip-images `
        --skip-existing `
        --dataset-name "VisDrone test-dev" `
        --modes yolo_sahi `
        2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    $ErrorActionPreference = $prevEap
}
