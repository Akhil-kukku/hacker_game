@echo off
echo 🧠 Installing Dependencies for Hacker Puzzle Game...
echo.

echo 📡 Installing Python packages...
cd backend
"C:\Users\X1\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install Python packages
    echo Try running: "C:\Users\X1\AppData\Local\Programs\Python\Python313\python.exe" -m pip install --user -r requirements.txt
    pause
    exit /b 1
)
echo ✅ Python packages installed successfully!

echo.
echo 🌐 Installing Node.js dependencies...
cd ..\frontend
"D:\clg\Node\npm.cmd" install
if %errorlevel% neq 0 (
    echo ❌ Failed to install Node.js dependencies
    pause
    exit /b 1
)
echo ✅ Node.js dependencies installed successfully!

echo.
echo 🎉 All dependencies installed successfully!
echo.
echo 🚀 You can now run the game:
echo 1. Double-click start_game.bat
echo 2. Or run: python start_game.py
echo 3. Open browser to: http://localhost:5173
echo.
pause
