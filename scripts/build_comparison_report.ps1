# Builds comparison_report/ outside outputs/
#   comparison_report/
#     KITTI report.txt
#     VisDrone report.txt
#     KITTI/   (5 pipeline folders)
#     VisDrone/

$Root = "C:\Users\khati\PycharmProjects\SAHI"
$Out  = Join-Path $Root "comparison_report"
$Work = Join-Path $Root "benchmark_working\supervisor_report"
if (-not (Test-Path $Work)) {
    $Work = Join-Path $Root "outputs\supervisor_report"
}

$Pipelines = @(
    @{ key = "yolo_only";      name = "yolo only" },
    @{ key = "yolo_sahi";      name = "yolo sahi" },
    @{ key = "sam3_only";      name = "sam3 only" },
    @{ key = "sam3_yolo";      name = "sam3 yolo" },
    @{ key = "yolo_sahi_sam3"; name = "yolo sahi sam3" }
)

$Datasets = @(
    @{ work = "kitti_val";          folder = "KITTI";    report = "KITTI report.txt";    images = 1496; label = "KITTI validation (1496 images)" },
    @{ work = "visdrone_test_dev";  folder = "VisDrone"; report = "VisDrone report.txt"; images = 1610; label = "VisDrone test dev (1610 images)" }
)

function Find-WorkDir($datasetWork, $modeKey) {
    $base = Join-Path $Work $datasetWork
    if (-not (Test-Path $base)) { return $null }
    foreach ($c in @($modeKey)) {
        $p = Join-Path $base $c
        if (Test-Path (Join-Path $p "metrics.json")) { return $p }
    }
    return $null
}

function Write-ConfusionMatrix($metricsPath, $destPath) {
    if (-not (Test-Path $metricsPath)) { return }
    $j = Get-Content $metricsPath -Raw | ConvertFrom-Json
    $tp = [int]$j.true_positives
    $fp = [int]$j.false_positives
    $fn = [int]$j.false_negatives
    $lines = @(
        "Detection confusion summary (IoU matched objects)",
        "",
        "Metric,Count",
        "True positives (correct detections),$tp",
        "False positives (wrong detections),$fp",
        "False negatives (missed objects),$fn",
        "",
        "Matrix,,Predicted yes,Predicted no",
        "Actual yes,,$tp,$fn",
        "Actual no,,$fp,"
    )
    $lines -join "`n" | Set-Content $destPath -Encoding UTF8
}

function Format-Num($n, $d = 3) {
    if ($null -eq $n) { return "n/a" }
    return ([math]::Round([double]$n, $d)).ToString()
}

function Build-DatasetReport($ds, $rows) {
    $W = 18
    $lines = @()
    $lines += "EagleEye AI Comparison Report"
    $lines += "Dataset: $($ds.label)"
    $lines += "Generated: $(Get-Date -Format 'yyyy MMM dd  HH:mm')"
    $lines += ""
    $lines += "All 5 pipelines"
    $lines += ""
    $hdr = "{0,-18} {1,7} {2,10} {3,8} {4,8} {5,8} {6,10} {7,7} {8,8}" -f `
        "Pipeline", "Images", "Precision", "Recall", "F1", "mAP50", "mAP50-95", "FPS", "Status"
    $lines += $hdr
    $lines += ("-" * 95)

    $complete = @()
    foreach ($r in $rows) {
        $lines += "{0,-18} {1,7} {2,10} {3,8} {4,8} {5,8} {6,10} {7,7} {8,8}" -f `
            $r.name,
            $r.images,
            (Format-Num $r.precision),
            (Format-Num $r.recall),
            (Format-Num $r.f1),
            (Format-Num $r.map50),
            (Format-Num $r.map50_95),
            (Format-Num $r.fps 2),
            $r.status
        if ($r.status -eq "complete") { $complete += $r }
    }

    $lines += ""
    $lines += "Dataset averages (complete pipelines only: $($complete.Count) of 5)"
    $lines += ""
    if ($complete.Count -gt 0) {
        $avg = [PSCustomObject]@{
            precision = ($complete | Measure-Object -Property precision -Average).Average
            recall    = ($complete | Measure-Object -Property recall -Average).Average
            f1        = ($complete | Measure-Object -Property f1 -Average).Average
            map50     = ($complete | Measure-Object -Property map50 -Average).Average
            map50_95  = ($complete | Measure-Object -Property map50_95 -Average).Average
            fps       = ($complete | Measure-Object -Property fps -Average).Average
            accuracy  = ($complete | Measure-Object -Property accuracy -Average).Average
        }
        $lines += "Metric              Average"
        $lines += "Precision           $(Format-Num $avg.precision)"
        $lines += "Recall              $(Format-Num $avg.recall)"
        $lines += "F1                  $(Format-Num $avg.f1)"
        $lines += "Accuracy            $(Format-Num $avg.accuracy)"
        $lines += "mAP50               $(Format-Num $avg.map50)"
        $lines += "mAP50-95            $(Format-Num $avg.map50_95)"
        $lines += "FPS                 $(Format-Num $avg.fps 2)"
    } else {
        $lines += "No pipeline has finished the full dataset yet."
    }

    $lines += ""
    $lines += "Per pipeline details are in:"
    $lines += "  comparison_report\$($ds.folder)\<pipeline name>\"
    $lines += "    metrics.json"
    $lines += "    results.csv"
    $lines += "    confusion_matrix.csv"
    return ($lines -join "`r`n")
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null

@(
    "EagleEye AI",
    "",
    "This folder has everything for your supervisor comparison.",
    "",
    "  KITTI report.txt       summary table + averages for KITTI",
    "  VisDrone report.txt    summary table + averages for VisDrone",
    "",
    "  KITTI/                 5 pipeline folders",
    "  VisDrone/              5 pipeline folders",
    "",
    "Each pipeline folder contains:",
    "  metrics.json           all numbers",
    "  results.csv            per image results",
    "  confusion_matrix.csv   TP / FP / FN matrix",
    "",
    "Refresh after benchmark progress:",
    "  .\scripts\build_comparison_report.ps1",
    ""
) -join "`r`n" | Set-Content (Join-Path $Out "README.txt") -Encoding UTF8

foreach ($ds in $Datasets) {
    $dsRoot = Join-Path $Out $ds.folder
    New-Item -ItemType Directory -Force -Path $dsRoot | Out-Null
    $reportRows = @()

    foreach ($p in $Pipelines) {
        $pipeDir = Join-Path $dsRoot $p.name
        New-Item -ItemType Directory -Force -Path $pipeDir | Out-Null

        $src = Find-WorkDir $ds.work $p.key
        if (-not $src) {
            @(
                "Pipeline: $($p.name)",
                "Status: not run yet",
                ""
            ) -join "`n" | Set-Content (Join-Path $pipeDir "metrics.json") -Encoding UTF8
            $reportRows += [PSCustomObject]@{
                name = $p.name; images = 0; precision = 0; recall = 0; f1 = 0
                map50 = 0; map50_95 = 0; fps = 0; accuracy = 0; status = "waiting"
            }
            continue
        }

        $metricsSrc = Join-Path $src "metrics.json"
        Copy-Item $metricsSrc (Join-Path $pipeDir "metrics.json") -Force
        $summary = Join-Path $src "summary.csv"
        if (Test-Path $summary) {
            Copy-Item $summary (Join-Path $pipeDir "results.csv") -Force
        } else {
            "image,note`n,no per image results yet" | Set-Content (Join-Path $pipeDir "results.csv") -Encoding UTF8
        }
        Write-ConfusionMatrix $metricsSrc (Join-Path $pipeDir "confusion_matrix.csv")

        $j = Get-Content $metricsSrc -Raw | ConvertFrom-Json
        $status = if ([int]$j.images_evaluated -ge $ds.images) { "complete" } else { "in progress" }
        $reportRows += [PSCustomObject]@{
            name      = $p.name
            images    = [int]$j.images_evaluated
            precision = [double]$j.precision
            recall    = [double]$j.recall
            f1        = [double]$j.f1
            accuracy  = [double]$j.accuracy
            map50     = [double]$j.map50
            map50_95  = [double]$j.map50_95
            fps       = [double]$j.fps
            status    = $status
        }
    }

    Build-DatasetReport $ds $reportRows | Set-Content (Join-Path $Out $ds.report) -Encoding UTF8
    Write-Host "Built $($ds.folder) and $($ds.report)"
}

Write-Host ""
Write-Host "Done: $Out"
