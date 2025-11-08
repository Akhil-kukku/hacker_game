# QUICKSTART: Evidence Collection & Metric Verification

## Complete Workflow (5 minutes)

### Step 1: Start the Server
```powershell
cd C:\Users\akhil\Downloads\PREV\hacker_game
python backend/main.py
```

Wait for:
- "Training ORDER engine on training_data.csv..."
- "Evaluating on test_data.csv..."
- "Uvicorn running on http://0.0.0.0:8000"

**Keep this terminal open!**

---

### Step 2: Open New Terminal & Navigate
```powershell
cd C:\Users\akhil\Downloads\PREV\hacker_game\evidence
```

---

### Step 3: Capture BASELINE (before adaptation)
```powershell
powershell -ExecutionPolicy Bypass -File capture_evidence.ps1
```

**Creates:** 
- `baseline.json`
- `baseline_order_status.json`
- `baseline_system_status.json`

---

### Step 4: Submit Labeled Feedback (trigger adaptation)
```powershell
powershell -ExecutionPolicy Bypass -File submit_feedback.ps1
```

**Result:** 5 labeled samples submitted (3 attacks, 2 benign)

---

### Step 5: Generate Load (test latency)
```powershell
powershell -ExecutionPolicy Bypass -File generate_load.ps1
```

**Repeat 3-5 times** to generate sufficient load for P95/P99 calculation.

---

### Step 6: Wait for Metrics Update
```powershell
Start-Sleep -Seconds 30
```

Give the system time to:
- Process feedback
- Re-evaluate metrics
- Calculate improvements

---

### Step 7: Capture CURRENT (after adaptation)
```powershell
powershell -ExecutionPolicy Bypass -File capture_evidence.ps1
```

**Creates:**
- `current.json`
- `current_order_status.json`
- `current_system_status.json`

---

### Step 8: Extract & Display Metrics
```powershell
powershell -ExecutionPolicy Bypass -File extract_metrics.ps1
```

**Output:**
- Detection Rate (target: 80%+)
- Accuracy Improvement (target: 12-18%)
- False Positive Reduction (target: 25%+)
- P95 Latency (target: <50ms)
- Confusion Matrix
- **VALUES TO UPDATE IN README.md**

---

## One-Line Complete Workflow

After server is running, paste this into PowerShell:

```powershell
cd C:\Users\akhil\Downloads\PREV\hacker_game\evidence; `
powershell -ExecutionPolicy Bypass -File capture_evidence.ps1; `
powershell -ExecutionPolicy Bypass -File submit_feedback.ps1; `
powershell -ExecutionPolicy Bypass -File generate_load.ps1; `
powershell -ExecutionPolicy Bypass -File generate_load.ps1; `
powershell -ExecutionPolicy Bypass -File generate_load.ps1; `
Start-Sleep -Seconds 30; `
powershell -ExecutionPolicy Bypass -File capture_evidence.ps1; `
powershell -ExecutionPolicy Bypass -File extract_metrics.ps1
```

---

## Files You'll Have

### Evidence Files (for archival/presentation):
```
evidence/
├── baseline.json                    # Initial evaluation metrics
├── baseline_order_status.json       # Initial ORDER engine state
├── baseline_system_status.json      # Initial system state
├── current.json                     # Post-adaptation metrics
├── current_order_status.json        # Current ORDER engine state
└── current_system_status.json       # Current system state
```

### Automation Scripts:
```
evidence/
├── capture_evidence.ps1             # Capture system state
├── submit_feedback.ps1              # Submit labeled samples
├── generate_load.ps1                # Generate latency test load
└── extract_metrics.ps1              # Parse & display evidence
```

---

## Update README.md

After running `extract_metrics.ps1`, you'll see:

```
VALUES TO UPDATE IN README.md:
Detection Rate:        85.2%
Accuracy Improvement:  14.3%
FP Reduction:          27.8%
P95 Latency:           42 ms
```

**Find & replace in README.md:**
- `[YOUR_DETECTION_RATE]%` → `85.2%` (your actual value)
- `[YOUR_ACCURACY_IMPROVEMENT]%` → `14.3%`
- `[YOUR_FP_REDUCTION]%` → `27.8%`
- `[YOUR_P95_LATENCY]ms` → `42ms`

---

## Troubleshooting

### "Connection refused"
**Fix:** Make sure server is running (`python backend/main.py`)

### "422 Unprocessable Content"
**Fix:** Server may not have finished training. Wait 10 seconds after startup.

### "No improvement data yet"
**Fix:** 
1. More feedback cycles needed (run `submit_feedback.ps1` multiple times)
2. Or model hasn't detected pattern differences yet

### Latency too high (>50ms)
**Fix:**
1. Run `generate_load.ps1` more times (need >100 samples for accurate P95)
2. Close other programs to reduce system load
3. P95 is more realistic than average - use that value

---

## What Each Script Does

| Script | Purpose | API Calls | Result |
|--------|---------|-----------|--------|
| `capture_evidence.ps1` | Save current metrics | `/order/evaluation-history`, `/order/status`, `/status` | 3 JSON files |
| `submit_feedback.ps1` | Submit labeled flows | `/order/feedback` (5 times) | Trigger adaptation |
| `generate_load.ps1` | Generate test traffic | `/flows` (200 flows) | Latency samples |
| `extract_metrics.ps1` | Parse & format evidence | None (reads local JSON) | Terminal report |

---

## Time Required

- **Initial Setup:** 2 minutes (install dependencies, generate data)
- **Baseline Capture:** 10 seconds
- **Adaptation:** 30 seconds (feedback + load generation)
- **Current Capture:** 10 seconds
- **Analysis:** 5 seconds

**Total:** ~5 minutes for complete evidence collection
