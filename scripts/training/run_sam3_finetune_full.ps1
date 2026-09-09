# Full SAM3 fine-tune: 5 epochs on all train images (VisDrone + KITTI).
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Sam3Dir = Join-Path $Root "external\sam3"
$LogRoot = Join-Path $Root "runs\sam3_finetune\logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Invoke-Train {
    param([string]$DatasetName)
    $tag = "full"
    $cfg = "configs/eagleeye/eagleeye_${DatasetName}_${tag}.yaml"
    $logFile = Join-Path $LogRoot "${DatasetName}_${tag}.log"
    Write-Host "`n=== TRAIN $DatasetName (5 epochs, full data) ===" -ForegroundColor Cyan
    Push-Location $Sam3Dir
    try {
        $env:PYTHONUNBUFFERED = "1"
        $env:USE_PERFLIB = "0"
        $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
        & $Py (Join-Path $Root "scripts\training\train_sam3_eagleeye.py") -c $cfg --use-cluster 0 --num-gpus 1 2>&1 |
            Tee-Object -FilePath $logFile
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $DatasetName (see $logFile)" }
    } finally {
        Pop-Location
    }
}

function Export-Weights {
    param([string]$DatasetName)
    $ckpt = Join-Path $Root "runs\sam3_finetune\$DatasetName\full\checkpoints\checkpoint.pt"
    & $Py (Join-Path $Root "scripts\training\export_sam3_finetune.py") `
        --dataset $DatasetName --checkpoint $ckpt
    if ($LASTEXITCODE -ne 0) { throw "Export failed: $DatasetName" }
}

foreach ($ds in @("visdrone", "kitti")) {
    & $Py (Join-Path $Root "scripts\training\prepare_sam3_coco.py") --dataset $ds
    & $Py (Join-Path $Root "scripts\training\generate_sam3_train_config.py") `
        --dataset $ds --epochs 5 --tag full
}

foreach ($ds in @("visdrone", "kitti")) {
    Invoke-Train -DatasetName $ds
    Export-Weights -DatasetName $ds
}

Write-Host "`n[DONE] Full 5-epoch SAM3 fine-tune complete (VisDrone + KITTI)." -ForegroundColor Green
