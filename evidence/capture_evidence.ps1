# Capture Evidence - Save current system state for analysis
# Run this after baseline and after adaptation to compare

$baseUrl = "http://localhost:8000"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  CAPTURING SYSTEM STATE FOR EVIDENCE" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if this is baseline or current capture
if (-not (Test-Path "baseline.json")) {
    $prefix = "baseline"
    Write-Host "This is your BASELINE capture (before adaptation)" -ForegroundColor Yellow
} else {
    $prefix = "current"
    Write-Host "This is your CURRENT capture (after adaptation)" -ForegroundColor Yellow
}

Write-Host ""

try {
    # Capture evaluation history
    Write-Host "Capturing evaluation history..." -ForegroundColor Gray
    $evalHistory = Invoke-RestMethod -Uri "$baseUrl/order/evaluation-history" -Method Get -ErrorAction Stop
    $evalHistory | ConvertTo-Json -Depth 10 | Out-File "$prefix.json" -Encoding UTF8
    Write-Host "  Saved: $prefix.json" -ForegroundColor Green
    
    # Capture ORDER engine status
    Write-Host "Capturing ORDER engine status..." -ForegroundColor Gray
    $orderStatus = Invoke-RestMethod -Uri "$baseUrl/order/status" -Method Get -ErrorAction Stop
    $orderStatus | ConvertTo-Json -Depth 10 | Out-File "${prefix}_order_status.json" -Encoding UTF8
    Write-Host "  Saved: ${prefix}_order_status.json" -ForegroundColor Green
    
    # Capture full system status
    Write-Host "Capturing system status..." -ForegroundColor Gray
    $systemStatus = Invoke-RestMethod -Uri "$baseUrl/status" -Method Get -ErrorAction Stop
    $systemStatus | ConvertTo-Json -Depth 10 | Out-File "${prefix}_system_status.json" -Encoding UTF8
    Write-Host "  Saved: ${prefix}_system_status.json" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "  SUCCESS: Evidence captured at $timestamp" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    if ($prefix -eq "baseline") {
        Write-Host "NEXT STEPS:" -ForegroundColor Yellow
        Write-Host "1. Run: powershell -ExecutionPolicy Bypass -File submit_feedback.ps1" -ForegroundColor White
        Write-Host "2. Run: powershell -ExecutionPolicy Bypass -File generate_load.ps1" -ForegroundColor White
        Write-Host "3. Wait 30 seconds for metrics to update" -ForegroundColor White
        Write-Host "4. Run this capture script again (will save as current.json)" -ForegroundColor White
        Write-Host "5. Run: powershell -ExecutionPolicy Bypass -File extract_metrics.ps1" -ForegroundColor White
    } else {
        Write-Host "READY FOR ANALYSIS!" -ForegroundColor Yellow
        Write-Host "Run: powershell -ExecutionPolicy Bypass -File extract_metrics.ps1" -ForegroundColor White
    }
    Write-Host ""
}
catch {
    Write-Host ""
    Write-Host "ERROR: Failed to capture evidence" -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Make sure the server is running at $baseUrl" -ForegroundColor Gray
    Write-Host "Start it with: python backend/main.py" -ForegroundColor Gray
    Write-Host ""
    exit 1
}
