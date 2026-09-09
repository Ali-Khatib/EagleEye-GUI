# EagleEye AI — benchmark progress (log is in benchmark_working, NOT outputs)
$Root = "C:\Users\khati\PycharmProjects\SAHI"
$Log  = Join-Path $Root "benchmark_working\supervisor_report\resume_2.log"
$Kit  = Join-Path $Root "benchmark_working\supervisor_report\kitti_val"

function Get-ModeProgress($modeKey) {
    $dir = Join-Path $Kit $modeKey
    $m = Join-Path $dir "metrics.json"
    if (-not (Test-Path $m)) { return $null }
    $j = Get-Content $m -Raw | ConvertFrom-Json
    [PSCustomObject]@{
        Mode = $j.mode; Images = $j.images_evaluated; F1 = [math]::Round($j.f1, 3)
        When = (Get-Item $m).LastWriteTime.ToString("HH:mm:ss")
    }
}

Clear-Host
Write-Host "Benchmark log: $Log`n"
while ($true) {
    Write-Host "=== $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Yellow
    $proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { $_.CommandLine -match "compare_all" }
    if ($proc) { Write-Host "RUNNING PID $($proc.ProcessId)" -ForegroundColor Green }
    else { Write-Host "NOT RUNNING" -ForegroundColor Red }
    foreach ($key in @("yolo_only","yolo_sahi","sam3_only","sam3_yolo","yolo_sahi_sam3")) {
        $r = Get-ModeProgress $key
        if ($r) {
            $st = if ($r.Images -ge 1496) { "DONE" } else { "RUN " }
            Write-Host ("  {0}  {1}/1496  F1={2}  {3}  {4}" -f $r.Mode, $r.Images, $r.F1, $r.When, $st)
        }
    }
    if (Test-Path $Log) {
        $hit = Get-Content $Log -Tail 3 -EA SilentlyContinue | Select-Object -Last 1
        Write-Host "  log: $hit"
    }
    Start-Sleep 15
}
