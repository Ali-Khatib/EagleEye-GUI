# EagleEye SAM3 fine-tune orchestrator: smoke test then full 15-epoch runs.
param(
    [ValidateSet("smoke", "full", "both")]
    [string]$Phase = "both",
    [ValidateSet("visdrone", "kitti", "all")]
    [string]$Dataset = "all",
    [switch]$SkipTrain
)

$ErrorActionPreference = "Continue"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Sam3Dir = Join-Path $Root "external\sam3"
$LogRoot = Join-Path $Root "runs\sam3_finetune\logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Invoke-Py {
    param([string[]]$Args)
    & $Py @Args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: python $($Args -join ' ')" }
}

function Invoke-Train {
    param(
        [string]$DatasetName,
        [string]$Tag
    )
    $cfg = "configs/eagleeye/eagleeye_${DatasetName}_${Tag}.yaml"
    $logFile = Join-Path $LogRoot "${DatasetName}_${Tag}.log"
    Write-Host "`n=== TRAIN $DatasetName ($Tag) ===" -ForegroundColor Cyan
    Write-Host "Config: $cfg"
    Write-Host "Log: $logFile"

    Push-Location $Sam3Dir
    try {
        $env:PYTHONUNBUFFERED = "1"
        $env:USE_PERFLIB = "0"
        & $Py (Join-Path $Root "scripts\training\train_sam3_eagleeye.py") -c $cfg --use-cluster 0 --num-gpus 1 2>&1 |
            Tee-Object -FilePath $logFile
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $DatasetName ($Tag)" }
    } finally {
        Pop-Location
    }
}

function Prepare-Dataset {
    param(
        [string]$DatasetName,
        [int]$Limit,
        [int]$Epochs,
        [string]$Tag
    )
    $limitArg = @()
    if ($Limit -gt 0) { $limitArg = @("--limit", "$Limit") }

    Write-Host "`n--- Prepare COCO: $DatasetName (limit=$Limit) ---" -ForegroundColor Yellow
    Invoke-Py @(
        (Join-Path $Root "scripts\training\prepare_sam3_coco.py"),
        "--dataset", $DatasetName
    ) + $limitArg

    Write-Host "--- Generate config: $DatasetName tag=$Tag epochs=$Epochs ---" -ForegroundColor Yellow
    Invoke-Py @(
        (Join-Path $Root "scripts\training\generate_sam3_train_config.py"),
        "--dataset", $DatasetName,
        "--epochs", "$Epochs",
        "--tag", $Tag
    ) + $(if ($Limit -gt 0) { @("--limit", "$Limit") } else { @() })
}

$datasets = if ($Dataset -eq "all") { @("visdrone", "kitti") } else { @($Dataset) }

if (-not (Test-Path $Sam3Dir)) {
    throw "Meta sam3 repo missing at $Sam3Dir — run: git clone https://github.com/facebookresearch/sam3.git external/sam3"
}
if (-not (Test-Path (Join-Path $Root "sam3.pt"))) {
    throw "Base checkpoint missing: $(Join-Path $Root 'sam3.pt')"
}

if ($Phase -in @("smoke", "both")) {
    foreach ($ds in $datasets) {
        Prepare-Dataset -DatasetName $ds -Limit 200 -Epochs 5 -Tag "smoke"
        if (-not $SkipTrain) {
            Invoke-Train -DatasetName $ds -Tag "smoke"
            Invoke-Py @(
                (Join-Path $Root "scripts\training\export_sam3_finetune.py"),
                "--dataset", $ds,
                "--checkpoint", (Join-Path $Root "runs\sam3_finetune\$ds\smoke\checkpoints\checkpoint.pt")
            )
        }
    }
}

if ($Phase -in @("full", "both")) {
    foreach ($ds in $datasets) {
        Prepare-Dataset -DatasetName $ds -Limit 0 -Epochs 15 -Tag "full"
        if (-not $SkipTrain) {
            Invoke-Train -DatasetName $ds -Tag "full"
            Invoke-Py @(
                (Join-Path $Root "scripts\training\export_sam3_finetune.py"),
                "--dataset", $ds,
                "--checkpoint", (Join-Path $Root "runs\sam3_finetune\$ds\full\checkpoints\checkpoint.pt")
            )
        }
    }
}

Write-Host "`n[DONE] SAM3 fine-tune pipeline finished." -ForegroundColor Green
