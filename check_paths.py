#!/usr/bin/env python3
"""
Path Checker for Hacker Puzzle Game
Verifies that Python and Node.js paths are correct
"""

import os
import subprocess
import platform

# User-specific paths
PYTHON_PATH = r"C:\Users\X1\AppData\Local\Programs\Python\Python313\python.exe"
NODE_PATH = r"D:\clg\Node\npm.cmd"

def check_python_path():
    """Check if Python path is correct and accessible"""
    print("🐍 Checking Python path...")
    print(f"   Path: {PYTHON_PATH}")
    
    if not os.path.exists(PYTHON_PATH):
        print("   ❌ Path does not exist!")
        return False
    
    try:
        # Try to run Python and get version
        result = subprocess.run([PYTHON_PATH, "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   ✅ Python found: {version}")
            return True
        else:
            print(f"   ❌ Python failed to run: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error running Python: {e}")
        return False

def check_node_path():
    """Check if Node.js path is correct and accessible"""
    print("\n📦 Checking Node.js path...")
    print(f"   Path: {NODE_PATH}")
    
    if not os.path.exists(NODE_PATH):
        print("   ❌ Path does not exist!")
        return False
    
    try:
        # Try to run npm and get version
        result = subprocess.run([NODE_PATH, "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   ✅ npm found: {version}")
            return True
        else:
            print(f"   ❌ npm failed to run: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error running npm: {e}")
        return False

def check_python_packages():
    """Check if required Python packages are installed"""
    print("\n📚 Checking Python packages...")
    
    packages = ["fastapi", "uvicorn", "deap"]
    all_installed = True
    
    for package in packages:
        try:
            result = subprocess.run([PYTHON_PATH, "-m", "pip", "show", package], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"   ✅ {package} - Installed")
            else:
                print(f"   ❌ {package} - Not installed")
                all_installed = False
        except Exception as e:
            print(f"   ❌ Error checking {package}: {e}")
            all_installed = False
    
    return all_installed

def check_node_modules():
    """Check if frontend dependencies are installed"""
    print("\n📦 Checking frontend dependencies...")
    
    if not os.path.exists("frontend/node_modules"):
        print("   ❌ node_modules not found - run 'npm install' in frontend directory")
        return False
    
    print("   ✅ node_modules found")
    return True

def main():
    """Main path check"""
    print("🧠 Hacker Puzzle Game - Path Checker")
    print("=" * 50)
    
    all_good = True
    
    # Check Python
    all_good &= check_python_path()
    
    # Check Node.js
    all_good &= check_node_path()
    
    # Check Python packages
    all_good &= check_python_packages()
    
    # Check frontend dependencies
    all_good &= check_node_modules()
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All paths and dependencies are correct!")
        print("\n🚀 You can now run the game using:")
        print("1. Double-click: start_game.bat")
        print("2. Or run: python start_game.py")
        print("3. Open browser to: http://localhost:5173")
    else:
        print("❌ Some issues were found.")
        print("\n🔧 To fix:")
        print("1. Install missing Python packages:")
        print(f"   {PYTHON_PATH} -m pip install -r backend/requirements.txt")
        print("2. Install frontend dependencies:")
        print("   cd frontend")
        print(f"   {NODE_PATH} install")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
