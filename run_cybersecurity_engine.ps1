# Self-Morphing AI Cybersecurity Engine - PowerShell Launcher
# Professional Cybersecurity Platform

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Self-Morphing AI Cybersecurity Engine" -ForegroundColor Green
Write-Host " Professional Cybersecurity Platform" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if required files exist
$requiredFiles = @(
    "backend/api_server.py",
    "backend/main_engine.py", 
    "backend/order_engine.py",
    "backend/chaos_engine.py"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "❌ ERROR: Missing required files:" -ForegroundColor Red
    foreach ($file in $missingFiles) {
        Write-Host "   - $file" -ForegroundColor Red
    }
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ All required files found" -ForegroundColor Green

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

# Install requirements if needed
if (Test-Path "requirements.txt") {
    Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow
    try {
        python -c "import fastapi, uvicorn, streamlit" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
            pip install -r requirements.txt
        }
    } catch {
        Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }
}

Write-Host ""
Write-Host "🚀 Starting Self-Morphing AI Cybersecurity Engine..." -ForegroundColor Green
Write-Host ""

# Start the launcher
try {
    python cybersecurity_launcher.py
} catch {
    Write-Host "❌ ERROR: Failed to start the cybersecurity engine" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "✅ Cybersecurity engine session ended" -ForegroundColor Green
Read-Host "Press Enter to exit"
