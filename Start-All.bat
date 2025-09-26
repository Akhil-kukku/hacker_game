@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d %~dp0

rem Create venv if missing
if not exist .venv (
  echo Creating virtual environment...
  py -3 -m venv .venv
)

call .venv\Scripts\activate.bat

rem Install dependencies
pip install --upgrade pip >nul 2>&1
pip install -r backend\requirements.txt

rem Create dirs
if not exist data mkdir data
if not exist models mkdir models
if not exist logs mkdir logs

rem Start system
python start_engine.py

endlocal
