#!/usr/bin/env python3
"""
Setup script for Self-Morphing AI Cybersecurity Engine
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        # Install from requirements.txt
        if os.path.exists("requirements.txt"):
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Requirements installed successfully")
        else:
            print("⚠️ requirements.txt not found, installing basic requirements...")
            basic_requirements = [
                "fastapi>=0.104.1",
                "uvicorn>=0.24.0", 
                "streamlit>=1.28.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "scikit-learn>=1.3.0",
                "requests>=2.31.0"
            ]
            
            for req in basic_requirements:
                subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
            
            print("✅ Basic requirements installed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "models", 
        "logs",
        "training_data",
        "training_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_launcher_shortcuts():
    """Create launcher shortcuts"""
    print("🔗 Creating launcher shortcuts...")
    
    system = platform.system().lower()
    
    if system == "windows":
        # Create batch file
        batch_content = '''@echo off
echo Starting Self-Morphing AI Cybersecurity Engine...
python cybersecurity_launcher.py
pause
'''
        with open("Start_Cybersecurity_Engine.bat", "w") as f:
            f.write(batch_content)
        print("✅ Created Start_Cybersecurity_Engine.bat")
        
    elif system == "darwin":  # macOS
        # Create shell script
        shell_content = '''#!/bin/bash
echo "Starting Self-Morphing AI Cybersecurity Engine..."
python3 cybersecurity_launcher.py
'''
        with open("start_cybersecurity_engine.sh", "w") as f:
            f.write(shell_content)
        os.chmod("start_cybersecurity_engine.sh", 0o755)
        print("✅ Created start_cybersecurity_engine.sh")
        
    else:  # Linux
        # Create shell script
        shell_content = '''#!/bin/bash
echo "Starting Self-Morphing AI Cybersecurity Engine..."
python3 cybersecurity_launcher.py
'''
        with open("start_cybersecurity_engine.sh", "w") as f:
            f.write(shell_content)
        os.chmod("start_cybersecurity_engine.sh", 0o755)
        print("✅ Created start_cybersecurity_engine.sh")

def verify_installation():
    """Verify installation"""
    print("🔍 Verifying installation...")
    
    # Check required files
    required_files = [
        "backend/api_server.py",
        "backend/main_engine.py",
        "backend/order_engine.py", 
        "backend/chaos_engine.py",
        "cybersecurity_launcher.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    
    # Test imports
    try:
        import fastapi
        import uvicorn
        import streamlit
        import pandas
        import numpy
        import sklearn
        import requests
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🛡️ Self-Morphing AI Cybersecurity Engine Setup")
    print("Professional Cybersecurity Platform")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python 3.8+ required")
        return False
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed: Could not install requirements")
        return False
    
    # Create directories
    create_directories()
    
    # Create launcher shortcuts
    create_launcher_shortcuts()
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Setup failed: Installation verification failed")
        return False
    
    print("\n✅ Setup completed successfully!")
    print("\n🚀 To start the cybersecurity engine:")
    print("   - Windows: Double-click 'Start_Cybersecurity_Engine.bat'")
    print("   - macOS/Linux: Run './start_cybersecurity_engine.sh'")
    print("   - Or run: python cybersecurity_launcher.py")
    
    print("\n📊 Access points:")
    print("   - Security Dashboard: http://localhost:8501")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - API Health Check: http://localhost:8000/health")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
