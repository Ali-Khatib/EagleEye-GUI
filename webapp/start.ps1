# Starts the FastAPI backend (port 8000) and the Vite frontend (port 5173)
# in two new PowerShell windows. Closes them when you press Ctrl+C here.
#
# Usage:
#   .\webapp\start.ps1
#
# Project: TUBITAK ARDEB 3501 - 124E099

$ErrorActionPreference = "Stop"

$root      = Split-Path -Parent $PSScriptRoot          # SAHI/
$backend   = Join-Path $PSScriptRoot "backend"
$frontend  = Join-Path $PSScriptRoot "frontend"
$venvAct   = Join-Path $root ".venv\Scripts\Activate.ps1"

Write-Host "[start] root     : $root"
Write-Host "[start] backend  : $backend"
Write-Host "[start] frontend : $frontend"

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "[start] First time: installing frontend deps (npm install)…" -ForegroundColor Yellow
    Push-Location $frontend
    npm install
    Pop-Location
}

# SAM 3 weights (optional — enables SAM 3 pipelines in the UI)
$sam3Pt = Join-Path $root "sam3.pt"
$sam3Env = "`$env:EXP_DEVICE = 'cuda'"
if (Test-Path $sam3Pt) {
    $sam3Env += "`n`$env:SAM3_WEIGHTS = '$sam3Pt'"
    Write-Host "[start] SAM3     : $sam3Pt" -ForegroundColor Green
} else {
    Write-Host "[start] SAM3     : not found (place sam3.pt at repo root or run scripts/download_sam3.py)" -ForegroundColor Yellow
}

# --- Spawn backend ---------------------------------------------------------
$backendCmd = @"
Set-Location '$backend';
if (Test-Path '$venvAct') { . '$venvAct' }
$sam3Env
Write-Host '[backend] starting FastAPI on http://127.0.0.1:8000' -ForegroundColor Cyan
python main.py
"@

$backendProc = Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", $backendCmd `
    -PassThru

# --- Spawn frontend --------------------------------------------------------
$frontendCmd = @"
Set-Location '$frontend';
Write-Host '[frontend] starting Vite on http://localhost:5173' -ForegroundColor Cyan
npm run dev
"@

$frontendProc = Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", $frontendCmd `
    -PassThru

Write-Host ""
Write-Host "[start] Backend  PID = $($backendProc.Id)"
Write-Host "[start] Frontend PID = $($frontendProc.Id)"
Write-Host ""
Write-Host "Open http://localhost:5173 in your browser."
Write-Host "Press Ctrl+C in this window to stop both."
Write-Host ""

try {
    while ($true) {
        Start-Sleep -Seconds 1
        if ($backendProc.HasExited) {
            Write-Host "[start] backend exited (code $($backendProc.ExitCode))" -ForegroundColor Yellow
            break
        }
        if ($frontendProc.HasExited) {
            Write-Host "[start] frontend exited (code $($frontendProc.ExitCode))" -ForegroundColor Yellow
            break
        }
    }
} finally {
    foreach ($p in @($backendProc, $frontendProc)) {
        if ($p -and -not $p.HasExited) {
            Write-Host "[start] stopping PID $($p.Id)…"
            try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch {}
        }
    }
    Write-Host "[start] done."
}
