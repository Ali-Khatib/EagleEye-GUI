# Chain SAM3 fine-tune: smoke (both datasets) -> full 5 epochs (both datasets).
$ErrorActionPreference = "Continue"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Sam3Dir = Join-Path $Root "external\sam3"
$LogRoot = Join-Path $Root "runs\sam3_finetune\logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Invoke-Train {
    param([string]$DatasetName, [string]$Tag)
    $cfg = "configs/eagleeye/eagleeye_${DatasetName}_${Tag}.yaml"
    $logFile = Join-Path $LogRoot "${DatasetName}_${Tag}.log"
    Write-Host "`n=== TRAIN $DatasetName ($Tag) ===" -ForegroundColor Cyan
    Push-Location $Sam3Dir
    try {
        $env:PYTHONUNBUFFERED = "1"
        $env:USE_PERFLIB = "0"
        $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
        & $Py (Join-Path $Root "scripts\training\train_sam3_eagleeye.py") -c $cfg --use-cluster 0 --num-gpus 1 2>&1 |
            Tee-Object -FilePath $logFile
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $DatasetName $Tag (see $logFile)" }
    } finally {
        Pop-Location
    }
}

function Export-Weights {
    param([string]$DatasetName, [string]$Tag)
    $ckpt = Join-Path $Root "runs\sam3_finetune\$DatasetName\$Tag\checkpoints\checkpoint.pt"
    & $Py (Join-Path $Root "scripts\training\export_sam3_finetune.py") `
        --dataset $DatasetName --checkpoint $ckpt
    if ($LASTEXITCODE -ne 0) { throw "Export failed: $DatasetName $Tag" }
}

# Smoke phase
foreach ($ds in @("visdrone", "kitti")) {
    Invoke-Train -DatasetName $ds -Tag "smoke"
    Export-Weights -DatasetName $ds -Tag "smoke"
}

# Full data prep + 5-epoch configs
foreach ($ds in @("visdrone", "kitti")) {
    & $Py (Join-Path $Root "scripts\training\prepare_sam3_coco.py") --dataset $ds
    if ($LASTEXITCODE -ne 0) { throw "COCO prep failed: $ds" }
    & $Py (Join-Path $Root "scripts\training\generate_sam3_train_config.py") `
        --dataset $ds --epochs 5 --tag full
    if ($LASTEXITCODE -ne 0) { throw "Config gen failed: $ds" }
}

# Full training
foreach ($ds in @("visdrone", "kitti")) {
    Invoke-Train -DatasetName $ds -Tag "full"
    Export-Weights -DatasetName $ds -Tag "full"
}

Write-Host "`n[DONE] Smoke + full SAM3 fine-tune complete." -ForegroundColor Green
