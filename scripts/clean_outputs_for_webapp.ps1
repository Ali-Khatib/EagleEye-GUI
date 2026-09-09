# Move benchmark clutter out of outputs/. Webapp uses outputs/webapp/ only.

$Root = "C:\Users\khati\PycharmProjects\SAHI"
$Out  = Join-Path $Root "outputs"
$Bench = Join-Path $Root "benchmark_working"

New-Item -ItemType Directory -Force -Path $Bench | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Out "webapp") | Out-Null

function Move-ToBench($name) {
    $src = Join-Path $Out $name
    $dst = Join-Path $Bench $name
    if (-not (Test-Path $src)) { return }
    $item = Get-Item $src -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        Remove-Item $src -Force -ErrorAction SilentlyContinue
        return
    }
    if (-not (Test-Path $dst)) {
        Move-Item $src $dst -Force
        Write-Host "Moved $name -> benchmark_working/"
    }
}

# supervisor_report lives ONLY in benchmark_working (never move into outputs)
$supBench = Join-Path $Bench "supervisor_report"
New-Item -ItemType Directory -Force -Path $supBench | Out-Null
$supOut = Join-Path $Out "supervisor_report"
if (Test-Path $supOut) {
    Remove-Item $supOut -Recurse -Force -ErrorAction SilentlyContinue
}

Move-ToBench "yolo_training_validation"
Move-ToBench "_archive"
Move-ToBench "webapp_demos"

foreach ($j in @("experiments", "validation")) {
    $p = Join-Path $Out $j
    if (Test-Path $p) { Remove-Item $p -Force -ErrorAction SilentlyContinue }
}

Remove-Item (Join-Path $Out "cleanup_outputs.ps1") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $Out "README.md") -Force -ErrorAction SilentlyContinue

$pipelines = @("yolo only", "yolo sahi", "sam3 only", "sam3 yolo", "yolo sahi sam3")
foreach ($ds in @("visdrone", "kitti", "stock")) {
    $dsDir = Join-Path $Out "webapp\$ds"
    New-Item -ItemType Directory -Force -Path $dsDir | Out-Null
    foreach ($p in $pipelines) {
        New-Item -ItemType Directory -Force -Path (Join-Path $dsDir $p) | Out-Null
    }
}

@(
    "WEBSITE OUTPUTS ONLY",
    "",
    "  webapp/visdrone/input_image.jpg",
    "  webapp/visdrone/yolo only/result.jpg",
    "  webapp/visdrone/yolo only/metrics.csv",
    "  webapp/visdrone/yolo only/notes.txt",
    "  (same for yolo sahi, sam3 only, sam3 yolo, yolo sahi sam3)",
    "  (same folders under kitti/ and stock/)",
    "",
    "Benchmark + comparison reports are NOT here:",
    "  benchmark_working/",
    "  comparison_report/",
    ""
) -join "`r`n" | Set-Content (Join-Path $Out "README.txt") -Encoding UTF8

Write-Host "outputs/webapp/ ready."
