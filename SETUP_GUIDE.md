# 🚀 Setup Guide for Hacker Puzzle Game

This guide will help you install all the necessary dependencies and get the game running on your system.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Memory**: At least 4GB RAM
- **Storage**: At least 500MB free space
- **Network**: Internet connection for downloading dependencies

## 🐍 Python Installation

### Windows
1. **Download Python**:
   - Go to [python.org/downloads](https://www.python.org/downloads/)
   - Download Python 3.8 or later (recommended: Python 3.11)
   - **IMPORTANT**: Check "Add Python to PATH" during installation

2. **Verify Installation**:
   ```cmd
   python --version
   ```
   Should show Python version (e.g., Python 3.11.0)

3. **Install pip** (if not included):
   ```cmd
   python -m ensurepip --upgrade
   ```

### macOS
1. **Using Homebrew** (recommended):
   ```bash
   brew install python
   ```

2. **Or download from python.org**:
   - Download and install from [python.org/downloads](https://www.python.org/downloads/)

3. **Verify Installation**:
   ```bash
   python3 --version
   ```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

## 📦 Node.js Installation

### Windows
1. **Download Node.js**:
   - Go to [nodejs.org](https://nodejs.org/)
   - Download LTS version (recommended)
   - Run the installer

2. **Verify Installation**:
   ```cmd
   node --version
   npm --version
   ```

### macOS
1. **Using Homebrew**:
   ```bash
   brew install node
   ```

2. **Or download from nodejs.org**:
   - Download and install from [nodejs.org](https://nodejs.org/)

### Linux (Ubuntu/Debian)
```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
```

## 🎮 Game Setup

### 1. Clone/Download the Project
Make sure you have the project files in a directory structure like this:
```
hacker_game/
├── backend/
│   ├── main.py
│   ├── game_logic.py
│   ├── levels.json
│   └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── src/
│   └── ...
├── README.md
└── start_game.bat (Windows) or start_game.py (Cross-platform)
```

### 2. Backend Setup
1. **Open Terminal/Command Prompt**
2. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: If you get permission errors, try:
   ```bash
   pip install --user -r requirements.txt
   ```

4. **Test the backend**:
   ```bash
   python main.py
   ```
   
   You should see output like:
   ```
   INFO:     Started server process [xxxx]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```

5. **Keep this terminal open** - the backend needs to keep running

### 3. Frontend Setup
1. **Open a NEW Terminal/Command Prompt**
2. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

3. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

4. **Start the frontend**:
   ```bash
   npm run dev
   ```
   
   You should see output like:
   ```
   VITE v5.x.x ready in xxx ms
   ➜  Local:   http://localhost:5173/
   ➜  Network: use --host to expose
   ```

5. **Keep this terminal open too**

### 4. Play the Game
1. **Open your web browser**
2. **Go to**: `http://localhost:5173`
3. **The game should load with a terminal interface**

## 🚀 Quick Start (Windows)

If you're on Windows, you can use the provided batch file:

1. **Double-click** `start_game.bat`
2. **Two command windows will open** - one for backend, one for frontend
3. **Wait for both to start** (you'll see success messages)
4. **Open your browser** to `http://localhost:5173`

## 🚀 Quick Start (Cross-platform)

Use the Python startup script:

1. **Open terminal/command prompt**
2. **Navigate to project root**:
   ```bash
   cd path/to/hacker_game
   ```

3. **Run the startup script**:
   ```bash
   python start_game.py
   ```

## 🐛 Troubleshooting

### Common Issues

#### Python Not Found
- **Windows**: Make sure you checked "Add Python to PATH" during installation
- **macOS/Linux**: Try `python3` instead of `python`

#### Port Already in Use
- **Backend (8000)**: Kill the process using port 8000
  ```bash
  # Windows
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  
  # macOS/Linux
  lsof -ti:8000 | xargs kill -9
  ```

- **Frontend (5173)**: Kill the process using port 5173
  ```bash
  # Windows
  netstat -ano | findstr :5173
  taskkill /PID <PID> /F
  
  # macOS/Linux
  lsof -ti:5173 | xargs kill -9
  ```

#### Dependencies Installation Failed
- **Python**: Try upgrading pip first:
  ```bash
  python -m pip install --upgrade pip
  ```

- **Node.js**: Try clearing npm cache:
  ```bash
  npm cache clean --force
  ```

#### CORS Errors
- Make sure backend is running on port 8000
- Check that frontend is connecting to `http://localhost:8000`
- Verify CORS middleware is enabled in backend

### Getting Help

1. **Check the logs** in both terminal windows for error messages
2. **Verify ports** are not in use by other applications
3. **Ensure all dependencies** are properly installed
4. **Check firewall settings** if you can't access localhost

## ✅ Verification

To verify everything is working:

1. **Backend**: Visit `http://localhost:8000` - should show health check message
2. **Frontend**: Visit `http://localhost:5173` - should show the game interface
3. **Game**: Type `help` in the terminal to see available commands

## 🎯 Next Steps

Once everything is running:
1. **Read the main README.md** for game instructions
2. **Type `list`** to see available missions
3. **Type `load 0`** to start the first level
4. **Have fun hacking!** 🎮💻

---

**Need more help?** Check the main README.md or create an issue in the project repository.
