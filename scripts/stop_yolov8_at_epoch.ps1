# Stop YOLOv8 training once target epoch is reached (resume may still display /100).
param(
    [int]$StopAt = 50,
    [string]$LogFile = ""
)

$root = Split-Path $PSScriptRoot -Parent
if (-not $LogFile) { $LogFile = Join-Path $root "benchmark_working\yolov8_train_resume.log" }
$status = Join-Path $root "benchmark_working\yolov8_stop_at_epoch.log"

function Log($m) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m"
    Add-Content $status $line
    Write-Host $line
}

Log "Watching log - stop training after epoch $StopAt"
$done = $false
while (-not $done) {
    if (Test-Path $LogFile) {
        $text = Get-Content $LogFile -Raw -ErrorAction SilentlyContinue
        $plain = $text -replace '\x1b\[[0-9;]*[A-Za-z]', ''
        $found = [regex]::Matches($plain, '(\d+)/(?:100|50)\s+[\d.]+G')
        if ($found.Count -gt 0) {
            $ep = [int]$found[$found.Count - 1].Groups[1].Value
            if ($ep -ge $StopAt) {
                Log "Epoch $ep reached (target $StopAt). Stopping training."
                Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                    Where-Object { $_.CommandLine -match "train_yolo_visdrone_v8" } |
                    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                Add-Content $LogFile "`n[DONE] Stopped at epoch $ep (target $StopAt)."
                $done = $true
            }
        }
    }
    if (-not $done) { Start-Sleep -Seconds 60 }
}
Log "Stop watcher finished."
