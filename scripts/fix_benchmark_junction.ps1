# Refresh junction after benchmark_working move (safe if already correct).
$supOut = "C:\Users\khati\PycharmProjects\SAHI\outputs\supervisor_report"
$supBench = "C:\Users\khati\PycharmProjects\SAHI\benchmark_working\supervisor_report"
if (-not (Test-Path $supBench)) { exit 0 }
if (Test-Path $supOut) {
    $item = Get-Item $supOut -Force
    if (-not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        Remove-Item $supOut -Recurse -Force -ErrorAction SilentlyContinue
    }
}
if (-not (Test-Path $supOut)) {
    cmd /c mklink /J "$supOut" "$supBench" | Out-Null
}
