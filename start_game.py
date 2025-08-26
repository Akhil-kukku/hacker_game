#!/usr/bin/env python3
"""
Hacker Puzzle Game Startup Script
Starts both backend and frontend servers
"""

import subprocess
import sys
import time
import os
import signal
import threading
import platform

# User-specific paths
PYTHON_PATH = r"C:\Users\X1\AppData\Local\Programs\Python\Python313\python.exe"
NODE_PATH = r"D:\clg\Node\npm.cmd"

def start_backend():
    """Start the backend server"""
    print("📡 Starting Backend Server...")
    try:
        os.chdir("backend")
        
        # Use specific Python path on Windows
        if platform.system() == "Windows":
            subprocess.run([PYTHON_PATH, "main.py"], check=True)
        else:
            subprocess.run([sys.executable, "main.py"], check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend failed to start: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")

def start_frontend():
    """Start the frontend server"""
    print("🌐 Starting Frontend Server...")
    try:
        os.chdir("frontend")
        
        # Use specific npm path on Windows
        if platform.system() == "Windows":
            subprocess.run([NODE_PATH, "run", "dev"], check=True)
        else:
            subprocess.run(["npm", "run", "dev"], check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend failed to start: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")

def main():
    print("🧠 Starting Hacker Puzzle Game...")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("backend") or not os.path.exists("frontend"):
        print("❌ Error: Please run this script from the project root directory")
        print("   Make sure 'backend' and 'frontend' folders exist")
        sys.exit(1)
    
    # Verify paths exist
    if platform.system() == "Windows":
        if not os.path.exists(PYTHON_PATH):
            print(f"❌ Error: Python not found at {PYTHON_PATH}")
            print("   Please update PYTHON_PATH in this script")
            sys.exit(1)
        if not os.path.exists(NODE_PATH):
            print(f"❌ Error: npm not found at {NODE_PATH}")
            print("   Please update NODE_PATH in this script")
            sys.exit(1)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a bit for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(3)
    
    # Start frontend
    start_frontend()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Game stopped by user")
        sys.exit(0)
