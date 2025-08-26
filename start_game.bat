@echo off
echo 🧠 Starting Hacker Puzzle Game...
echo.

echo 📡 Starting Backend Server...
start "Backend Server" cmd /k "cd backend && "C:\Users\X1\AppData\Local\Programs\Python\Python313\python.exe" main.py"

echo ⏳ Waiting for backend to start...
timeout /t 3 /nobreak >nul

echo 🌐 Starting Frontend Server...
start "Frontend Server" cmd /k "cd frontend && "D:\clg\Node\npm.cmd" run dev"

echo.
echo 🎮 Game is starting up!
echo 📡 Backend: http://localhost:8000
echo 🌐 Frontend: http://localhost:5173
echo.
echo Press any key to close this window...
pause >nul
