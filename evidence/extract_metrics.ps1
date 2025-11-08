# Extract and Display Performance Metrics
# This script parses captured JSON files and presents key metrics

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  CYBERSECURITY ENGINE - PERFORMANCE EVIDENCE REPORT" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if files exist
$requiredFiles = @("baseline.json", "current.json", "current_order_status.json")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "ERROR: Missing required files:" -ForegroundColor Red
    foreach ($file in $missingFiles) {
        Write-Host "  - $file" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Please run capture_evidence.ps1 to collect evidence files." -ForegroundColor Gray
    Write-Host ""
    Write-Host "Quick workflow:" -ForegroundColor Cyan
    Write-Host "  1. powershell -ExecutionPolicy Bypass -File capture_evidence.ps1" -ForegroundColor White
    Write-Host "  2. powershell -ExecutionPolicy Bypass -File submit_feedback.ps1" -ForegroundColor White
    Write-Host "  3. powershell -ExecutionPolicy Bypass -File generate_load.ps1" -ForegroundColor White
    Write-Host "  4. powershell -ExecutionPolicy Bypass -File capture_evidence.ps1" -ForegroundColor White
    Write-Host "  5. powershell -ExecutionPolicy Bypass -File extract_metrics.ps1" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Load data
try {
    $baseline = Get-Content baseline.json -Raw | ConvertFrom-Json
    $current = Get-Content current.json -Raw | ConvertFrom-Json
    $orderStatus = Get-Content current_order_status.json -Raw | ConvertFrom-Json
}
catch {
    Write-Host "ERROR: Failed to parse JSON files" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Yellow
    exit 1
}

# Section 1: Detection Rate
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host " 1. DETECTION RATE (TP / (TP + FN))" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray

$baselineDetection = 0
$currentDetection = 0

if ($baseline.history -and $baseline.history.Count -gt 0) {
    $baselineDetection = $baseline.history[0].detection_rate
}

if ($current.history -and $current.history.Count -gt 0) {
    $currentDetection = $current.history[-1].detection_rate
}

Write-Host "   Baseline:  $([math]::Round($baselineDetection * 100, 2))%" -ForegroundColor Gray
Write-Host "   Current:   $([math]::Round($currentDetection * 100, 2))%" -ForegroundColor Green -NoNewline
if ($currentDetection -ge 0.80) {
    Write-Host "  [PASS] MEETS 80 percent CLAIM" -ForegroundColor Cyan
} else {
    Write-Host "  (Target: 80 percent)" -ForegroundColor Yellow
}
Write-Host ""

# Section 2: Accuracy Improvement
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host " 2. ACCURACY IMPROVEMENT" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray

if ($current.summary -and $current.summary.accuracy_improvement_percent) {
    $accuracyGain = [math]::Round($current.summary.accuracy_improvement_percent, 2)
    $baselineAcc = [math]::Round($current.summary.baseline.accuracy * 100, 2)
    $currentAcc = [math]::Round($current.summary.latest.accuracy * 100, 2)
    
    Write-Host "   Baseline Accuracy:  $baselineAcc%" -ForegroundColor Gray
    Write-Host "   Current Accuracy:   $currentAcc%" -ForegroundColor Gray
    Write-Host "   Improvement:        +$accuracyGain%" -ForegroundColor Green -NoNewline
    
    if ($accuracyGain -ge 12 -and $accuracyGain -le 18) {
        Write-Host "  [PASS] IN TARGET RANGE (12-18 percent)" -ForegroundColor Cyan
    } elseif ($accuracyGain -gt 0) {
        Write-Host "  (Target: 12-18 percent)" -ForegroundColor Yellow
    }
    Write-Host ""
} else {
    Write-Host "   No improvement data yet (need more adaptation cycles)" -ForegroundColor Yellow
    Write-Host ""
}

# Section 3: False Positive Reduction
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host " 3. FALSE POSITIVE REDUCTION" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray

if ($current.summary -and $current.summary.false_positive_reduction_percent) {
    $fpReduction = [math]::Round($current.summary.false_positive_reduction_percent, 2)
    $baselineFP = [math]::Round($current.summary.baseline.false_positive_rate * 100, 2)
    $currentFP = [math]::Round($current.summary.latest.false_positive_rate * 100, 2)
    
    Write-Host "   Baseline FP Rate:  $baselineFP%" -ForegroundColor Gray
    Write-Host "   Current FP Rate:   $currentFP%" -ForegroundColor Gray
    Write-Host "   Reduction:         -$fpReduction%" -ForegroundColor Green -NoNewline
    
    if ($fpReduction -ge 25) {
        Write-Host "  [PASS] MEETS 25 percent CLAIM" -ForegroundColor Cyan
    } elseif ($fpReduction -gt 0) {
        Write-Host "  (Target: 25 percent)" -ForegroundColor Yellow
    }
    Write-Host ""
} else {
    Write-Host "   No reduction data yet (need more adaptation cycles)" -ForegroundColor Yellow
    Write-Host ""
}

# Section 4: Processing Latency
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host " 4. PROCESSING LATENCY (per flow)" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray

$avgLatency = [math]::Round($orderStatus.performance_metrics.avg_processing_time_ms, 2)
$p50 = [math]::Round($orderStatus.performance_metrics.latency_p50_ms, 2)
$p95 = [math]::Round($orderStatus.performance_metrics.latency_p95_ms, 2)
$p99 = [math]::Round($orderStatus.performance_metrics.latency_p99_ms, 2)

Write-Host "   Average:   $avgLatency ms" -ForegroundColor Gray
Write-Host "   P50:       $p50 ms" -ForegroundColor Gray
Write-Host "   P95:       $p95 ms" -ForegroundColor Green -NoNewline
if ($p95 -lt 50) {
    Write-Host "  [PASS] MEETS 50ms CLAIM" -ForegroundColor Cyan
} else {
    Write-Host "  (Target: under 50ms)" -ForegroundColor Yellow
}
Write-Host "   P99:       $p99 ms" -ForegroundColor Gray
Write-Host ""

# Section 5: Confusion Matrix
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host " 5. CONFUSION MATRIX (Latest Evaluation)" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray

if ($current.history -and $current.history.Count -gt 0) {
    $latest = $current.history[-1]
    
    Write-Host "   True Positives:   $($latest.tp)" -ForegroundColor Green
    Write-Host "   False Positives:  $($latest.fp)" -ForegroundColor Red
    Write-Host "   True Negatives:   $($latest.tn)" -ForegroundColor Green
    Write-Host "   False Negatives:  $($latest.fn)" -ForegroundColor Red
    Write-Host ""
    Write-Host "   Precision:        $([math]::Round($latest.precision * 100, 2))%" -ForegroundColor Cyan
    Write-Host "   Recall:           $([math]::Round($latest.recall * 100, 2))%" -ForegroundColor Cyan
    Write-Host "   F1-Score:         $([math]::Round($latest.f1 * 100, 2))%" -ForegroundColor Cyan
    Write-Host ""
}

# Section 6: System Statistics
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host " 6. SYSTEM PERFORMANCE STATS" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray

Write-Host "   Total Flows Processed:   $($orderStatus.performance_metrics.total_flows_processed)" -ForegroundColor Gray
Write-Host "   Batches Processed:       $($orderStatus.performance_metrics.batches_processed)" -ForegroundColor Gray
Write-Host "   Anomalies Detected:      $($orderStatus.performance_metrics.anomalies_detected)" -ForegroundColor Gray
Write-Host "   Model Trained:           $($orderStatus.is_trained)" -ForegroundColor Gray
Write-Host "   Model Type:              $($orderStatus.model_type)" -ForegroundColor Gray
Write-Host ""

# Summary
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  EVIDENCE FILES SAVED IN: $(Get-Location)" -ForegroundColor Cyan
Write-Host "  - baseline.json, current.json, current_order_status.json" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Output values for README update
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host " VALUES TO UPDATE IN README.md:" -ForegroundColor Magenta
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "Detection Rate:        $([math]::Round($currentDetection * 100, 2))%"
if ($current.summary -and $current.summary.accuracy_improvement_percent) {
    Write-Host "Accuracy Improvement:  $([math]::Round($current.summary.accuracy_improvement_percent, 2))%"
}
if ($current.summary -and $current.summary.false_positive_reduction_percent) {
    Write-Host "FP Reduction:          $([math]::Round($current.summary.false_positive_reduction_percent, 2))%"
}
Write-Host "P95 Latency:           $p95 ms"
Write-Host ""
