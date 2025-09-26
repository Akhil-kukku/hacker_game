#!/usr/bin/env python3
"""
Self-Morphing AI Cybersecurity Engine - Startup Script
Launches the complete system with all components
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'numpy', 'pandas', 'sklearn', 
        'joblib', 'deap', 'streamlit', 'requests', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing dependencies: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r backend/requirements.txt")
        return False
    
    return True

def create_directories():
    """Create required directories"""
    directories = ['data', 'models', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")

def start_api_server():
    """Start the API server"""
    try:
        print("🚀 Starting API Server...")
        env = os.environ.copy()
        # Favor production logging level unless overridden
        env.setdefault('UVICORN_LOG_LEVEL', 'info')
        process = subprocess.Popen([
            sys.executable, 'backend/api_server.py'
        ], cwd=os.getcwd(), env=env)
        return process
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def start_dashboard():
    """Start the Streamlit dashboard"""
    try:
        print("📊 Starting Dashboard...")
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'backend/dashboard.py',
            '--server.port', '8501',
            '--server.headless', 'true'
        ], cwd=os.getcwd())
        return process
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return None

def wait_for_api():
    """Wait for API server to be ready"""
    import requests
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get('http://localhost:8000/health', timeout=2)
            if response.status_code == 200:
                print("✅ API Server is ready")
                return True
        except:
            pass
        time.sleep(1)
        if attempt % 5 == 0:
            print(f"⏳ Waiting for API server... ({attempt + 1}/{max_attempts})")
    
    print("❌ API server failed to start")
    return False

def main():
    """Main startup function"""
    print("🛡️ Self-Morphing AI Cybersecurity Engine v2.1")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        sys.exit(1)
    
    # Wait for API to be ready
    if not wait_for_api():
        api_process.terminate()
        sys.exit(1)
    
    # Start dashboard
    dashboard_process = start_dashboard()
    if not dashboard_process:
        api_process.terminate()
        sys.exit(1)
    
    print("\n🎉 System started successfully!")
    print("\n📊 Access Points:")
    print("   Dashboard: http://localhost:8501")
    print("   API Docs:  http://localhost:8000/docs")
    print("   Health:    http://localhost:8000/health")
    print("\n⏹️  Press Ctrl+C to stop all services")
    
    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down...")
        if api_process:
            api_process.terminate()
        if dashboard_process:
            dashboard_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            # Check if processes are still running
            if api_process.poll() is not None:
                print("❌ API server stopped unexpectedly")
                break
            if dashboard_process.poll() is not None:
                print("❌ Dashboard stopped unexpectedly")
                break
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    signal_handler(None, None)

if __name__ == "__main__":
    main()
