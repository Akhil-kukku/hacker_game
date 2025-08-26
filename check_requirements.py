#!/usr/bin/env python3
"""
Requirements Checker for Hacker Puzzle Game
Checks if all necessary dependencies are installed
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} - Installed")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not installed")
        return False

def check_node_version():
    """Check if Node.js is installed and version is compatible"""
    print("\n📦 Checking Node.js...")
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Node.js {version} - Installed")
            
            # Check if version is 16+
            version_num = version.replace('v', '').split('.')[0]
            if int(version_num) >= 16:
                print(f"✅ Node.js version {version_num} - Compatible")
                return True
            else:
                print(f"⚠️ Node.js version {version_num} - Need version 16+")
                return False
        else:
            print("❌ Node.js - Not installed or not accessible")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Node.js - Not installed or not accessible")
        return False

def check_npm():
    """Check if npm is available"""
    print("\n📦 Checking npm...")
    try:
        result = subprocess.run(['npm', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ npm {version} - Installed")
            return True
        else:
            print("❌ npm - Not installed or not accessible")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ npm - Not installed or not accessible")
        return False

def check_ports():
    """Check if required ports are available"""
    print("\n🔌 Checking port availability...")
    
    import socket
    
    def check_port(port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.close()
            return True
        except OSError:
            return False
    
    ports = [8000, 5173]  # Backend and frontend ports
    
    for port in ports:
        if check_port(port):
            print(f"✅ Port {port} - Available")
        else:
            print(f"⚠️ Port {port} - In use (may need to kill existing process)")
    
    return True

def main():
    """Main requirements check"""
    print("🧠 Hacker Puzzle Game - Requirements Checker")
    print("=" * 50)
    
    all_good = True
    
    # Check Python
    all_good &= check_python_version()
    
    print("\n📚 Checking Python packages...")
    all_good &= check_python_package("fastapi", "fastapi")
    all_good &= check_python_package("uvicorn", "uvicorn")
    all_good &= check_python_package("deap", "deap")
    
    # Check Node.js
    all_good &= check_node_version()
    all_good &= check_npm()
    
    # Check ports
    check_ports()
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All requirements are met! You can run the game.")
        print("\n🚀 To start the game:")
        print("1. Run: python start_game.py")
        print("2. Or on Windows: double-click start_game.bat")
        print("3. Open browser to: http://localhost:5173")
    else:
        print("❌ Some requirements are missing.")
        print("\n📚 Please follow the SETUP_GUIDE.md instructions to install missing dependencies.")
        print("\nMissing items:")
        if not check_python_version():
            print("- Python 3.8+")
        if not check_node_version():
            print("- Node.js 16+")
        if not check_npm():
            print("- npm")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
