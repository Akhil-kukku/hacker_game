@echo off
echo.
echo ========================================
echo  Self-Morphing AI Cybersecurity Engine
echo  Professional Cybersecurity Platform
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Checking Python installation...
python --version

echo.
echo Starting Self-Morphing AI Cybersecurity Engine...
echo.

REM Start the main launcher
python cybersecurity_launcher.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start the cybersecurity engine
    echo Please check the error messages above
    pause
)


