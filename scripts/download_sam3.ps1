# Download sam3.pt into the SAHI repo root (Windows-friendly paths).
# Usage:
#   .\scripts\download_sam3.ps1
#   .\scripts\download_sam3.ps1 -Token "hf_...."

param(
    [string]$Token = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "sam3.pt"

# Prefer py launcher, then common Python installs
$python = $null
foreach ($candidate in @(
    "py",
    "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe",
    "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe",
    "$env:LOCALAPPDATA\Python\bin\python.exe",
    "python"
)) {
    if ($candidate -eq "py") {
        if (Get-Command py -ErrorAction SilentlyContinue) { $python = "py"; break }
    } elseif (Test-Path $candidate) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    Write-Host "[ERROR] Python not found. Install Python 3.10+ or use PyCharm's interpreter." -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Python: $python"
Write-Host "[INFO] Target: $dest"

& $python -m pip install -q huggingface_hub

if ($Token) {
    $env:HF_TOKEN = $Token
    Write-Host "[INFO] Using HF_TOKEN from -Token argument."
} elseif (-not $env:HF_TOKEN) {
    Write-Host ""
    Write-Host "Hugging Face token required (https://huggingface.co/settings/tokens):" -ForegroundColor Yellow
    Write-Host "  `$env:HF_TOKEN = 'hf_....'" -ForegroundColor Cyan
    Write-Host "  Or: .\scripts\download_sam3.ps1 -Token hf_...." -ForegroundColor Cyan
    Write-Host "  Or: & `$python -c `"from huggingface_hub import login; login()`"" -ForegroundColor Cyan
    Write-Host ""
}

Set-Location $root
& $python scripts/download_sam3.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-Path $dest) {
    $mb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
    Write-Host "[OK] Ready: $dest ($mb MB)" -ForegroundColor Green
    Write-Host "Restart webapp: .\webapp\start.ps1"
} else {
    Write-Host "[WARN] sam3.pt still missing at $dest" -ForegroundColor Yellow
    exit 1
}
