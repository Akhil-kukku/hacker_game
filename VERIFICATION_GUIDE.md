# 🔬 Performance Metrics Verification Guide

## Quick Start (5 Minutes)

### 1. Install & Generate Data
```powershell
cd C:\Users\akhil\Downloads\PREV\hacker_game
cd backend
pip install -r requirements.txt
cd ..\tools
python generate_sample_dataset.py
cd ..
mkdir evidence
```

### 2. Start Server
```powershell
cd backend
python api_server.py
# Wait for "✅ ORDER engine trained successfully" message
```

### 3. Capture Baseline (New Terminal)
```powershell
cd C:\Users\akhil\Downloads\PREV\hacker_game\evidence
curl http://localhost:8000/order/evaluation-history | Out-File -Encoding utf8 baseline.json
curl http://localhost:8000/order/status | Out-File -Encoding utf8 baseline_status.json
```

### 4. Submit Feedback & Generate Load
```powershell
# Create and run submit_feedback.ps1 (see full script below)
powershell -ExecutionPolicy Bypass -File submit_feedback.ps1

# Create and run generate_load.ps1 (see full script below)
powershell -ExecutionPolicy Bypass -File generate_load.ps1
```

### 5. Capture Current Metrics
```powershell
curl http://localhost:8000/order/evaluation-history | Out-File -Encoding utf8 current.json
curl http://localhost:8000/order/status | Out-File -Encoding utf8 current_status.json
Copy-Item ..\backend\data\metrics_history.json -Destination metrics_history.json
```

### 6. Extract Evidence
```powershell
# Run extract_metrics.ps1 (see full script below)
powershell -ExecutionPolicy Bypass -File extract_metrics.ps1
```

---

## 📄 Required Scripts

### submit_feedback.ps1
```powershell
$feedbackSamples = @(
    @{ src_ip="203.0.113.45"; dst_ip="10.0.0.8"; src_port=55123; dst_port=22; protocol="TCP"; packet_count=2500; byte_count=1800000; duration=0.3; flags="SYN"; is_attack=$true },
    @{ src_ip="203.0.114.78"; dst_ip="10.0.0.12"; src_port=48901; dst_port=3389; protocol="TCP"; packet_count=3000; byte_count=2100000; duration=0.2; flags="PSH"; is_attack=$true },
    @{ src_ip="192.168.1.55"; dst_ip="10.0.0.5"; src_port=54321; dst_port=443; protocol="HTTPS"; packet_count=150; byte_count=45000; duration=2.5; flags="ACK"; is_attack=$false },
    @{ src_ip="192.168.1.88"; dst_ip="10.0.0.9"; src_port=60123; dst_port=80; protocol="HTTP"; packet_count=80; byte_count=12000; duration=1.8; flags="FIN"; is_attack=$false },
    @{ src_ip="203.0.115.99"; dst_ip="10.0.0.20"; src_port=40001; dst_port=1433; protocol="TCP"; packet_count=5000; byte_count=3500000; duration=0.1; flags="SYN"; is_attack=$true }
)

foreach ($sample in $feedbackSamples) {
    $body = $sample | ConvertTo-Json
    Invoke-RestMethod -Uri "http://localhost:8000/order/feedback" -Method POST -Body $body -ContentType "application/json"
    Write-Host "Submitted: $($sample.src_ip) -> $($sample.dst_ip) [Attack: $($sample.is_attack)]"
    Start-Sleep -Milliseconds 500
}
Write-Host "Feedback complete. Model adapting..."
```

### generate_load.ps1
```powershell
$flows = @()
for ($i = 1; $i -le 200; $i++) {
    $flows += @{
        src_ip = "192.168.1.$((Get-Random -Minimum 2 -Maximum 254))"
        dst_ip = "10.0.0.$((Get-Random -Minimum 2 -Maximum 254))"
        src_port = Get-Random -Minimum 1024 -Maximum 65535
        dst_port = @(80, 443, 22, 53, 25, 3306)[$(Get-Random -Minimum 0 -Maximum 6)]
        protocol = @("TCP", "UDP", "HTTP", "HTTPS")[$(Get-Random -Minimum 0 -Maximum 4)]
        packet_count = Get-Random -Minimum 10 -Maximum 1000
        byte_count = Get-Random -Minimum 500 -Maximum 50000
        duration = [math]::Round((Get-Random -Minimum 50 -Maximum 2000) / 1000, 3)
        flags = @("SYN", "ACK", "FIN", "PSH", "RST", "")[$(Get-Random -Minimum 0 -Maximum 6)]
    }
}
$body = $flows | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/flows" -Method POST -Body $body -ContentType "application/json"
Write-Host "Sent 200 flows"
```

### extract_metrics.ps1
```powershell
$baseline = Get-Content baseline.json | ConvertFrom-Json
$current = Get-Content current.json | ConvertFrom-Json
$orderStatus = Get-Content current_status.json | ConvertFrom-Json

Write-Host "=== PERFORMANCE EVIDENCE ===" -ForegroundColor Cyan
Write-Host ""

# Detection Rate
$baselineDetection = if ($baseline.history.Count -gt 0) { $baseline.history[0].detection_rate } else { 0 }
$currentDetection = if ($current.history.Count -gt 0) { $current.history[-1].detection_rate } else { 0 }
Write-Host "DETECTION RATE: $([math]::Round($currentDetection * 100, 2))%" -ForegroundColor Green
Write-Host "  (Baseline: $([math]::Round($baselineDetection * 100, 2))%)"
Write-Host ""

# Accuracy Improvement
if ($current.summary.accuracy_improvement_percent) {
    Write-Host "ACCURACY IMPROVEMENT: +$([math]::Round($current.summary.accuracy_improvement_percent, 2))%" -ForegroundColor Green
    Write-Host "  Baseline: $([math]::Round($current.summary.baseline.accuracy * 100, 2))%"
    Write-Host "  Current:  $([math]::Round($current.summary.latest.accuracy * 100, 2))%"
    Write-Host ""
}

# False Positive Reduction
if ($current.summary.false_positive_reduction_percent) {
    Write-Host "FALSE POSITIVE REDUCTION: -$([math]::Round($current.summary.false_positive_reduction_percent, 2))%" -ForegroundColor Green
    Write-Host "  Baseline FP: $([math]::Round($current.summary.baseline.false_positive_rate * 100, 2))%"
    Write-Host "  Current FP:  $([math]::Round($current.summary.latest.false_positive_rate * 100, 2))%"
    Write-Host ""
}

# Latency
Write-Host "LATENCY (P95): $([math]::Round($orderStatus.performance_metrics.latency_p95_ms, 2))ms" -ForegroundColor Green
Write-Host "  Average: $([math]::Round($orderStatus.performance_metrics.avg_processing_time_ms, 2))ms"
Write-Host "  P50:     $([math]::Round($orderStatus.performance_metrics.latency_p50_ms, 2))ms"
Write-Host "  P99:     $([math]::Round($orderStatus.performance_metrics.latency_p99_ms, 2))ms"
Write-Host ""

# Confusion Matrix
if ($current.history.Count -gt 0) {
    $latest = $current.history[-1]
    Write-Host "CONFUSION MATRIX:"
    Write-Host "  TP: $($latest.tp)  FP: $($latest.fp)"
    Write-Host "  FN: $($latest.fn)  TN: $($latest.tn)"
    Write-Host "  Precision: $([math]::Round($latest.precision * 100, 2))%"
    Write-Host "  Recall:    $([math]::Round($latest.recall * 100, 2))%"
    Write-Host "  F1-Score:  $([math]::Round($latest.f1 * 100, 2))%"
}

Write-Host ""
Write-Host "Evidence saved in: $(Get-Location)" -ForegroundColor Cyan
```

---

## 📊 How to Present Results

### For Documentation/README
Replace placeholder values with YOUR numbers:

```markdown
- **[X]% detection rate** - from current.json: `history[-1].detection_rate * 100`
- **[Y]% accuracy improvement** - from current.json: `summary.accuracy_improvement_percent`
- **[Z]% false positive reduction** - from current.json: `summary.false_positive_reduction_percent`
- **<[W]ms processing** - from current_status.json: `performance_metrics.latency_p95_ms`
```

### For Presentations
Use screenshots of:
1. Terminal output from `extract_metrics.ps1`
2. Confusion matrix table
3. API responses (current.json formatted)

### For Academic/Technical Reports
Include:
- `baseline.json` - Initial evaluation
- `current.json` - Post-adaptation evaluation
- `metrics_history.json` - Longitudinal data
- Server logs showing "Applying feedback update" and "Periodic evaluation"

---

## 🔧 Troubleshooting

### Server won't start
```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000
# Kill the process if needed
taskkill /PID <PID> /F
```

### No improvement showing
- Run feedback script multiple times
- Wait 15+ minutes for periodic evaluation
- Submit more diverse feedback (mix of attacks and normal traffic)

### Latency too high
- Reduce flow count in generate_load.ps1 (200 → 50)
- Run load script fewer times
- Check CPU usage (should be <30%)

---

## 📁 Evidence Files Created

After running all steps, you'll have in `evidence/`:

```
baseline.json                 - Initial metrics
baseline_status.json          - Initial system state
current.json                  - Post-adaptation metrics
current_status.json           - Current system state
metrics_history.json          - Full evaluation history
evaluation_result.json        - Manual evaluation trigger result
```

**Archive these for proof of performance claims!**

---

## ✅ Validation Checklist

- [ ] Training data generated (12,399 samples)
- [ ] Server started and trained successfully
- [ ] Baseline captured
- [ ] Feedback submitted (5+ samples)
- [ ] Load generated (200+ flows, 3+ times)
- [ ] Current metrics captured
- [ ] Metrics extracted and reviewed
- [ ] README.md updated with actual values
- [ ] Evidence files archived

---

**Last Updated:** November 2025  
**Version:** 2.0
