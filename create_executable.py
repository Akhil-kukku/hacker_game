#!/usr/bin/env python3
"""
Create Executable for Self-Morphing AI Cybersecurity Engine
Creates a standalone executable that runs the entire system
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

def create_executable():
    """Create executable using PyInstaller"""
    print("🛡️ Creating Self-Morphing AI Cybersecurity Engine Executable")
    print("=" * 70)
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print("✅ PyInstaller is available")
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✅ PyInstaller installed")
    
    # Create main launcher script
    launcher_script = create_launcher_script()
    
    # Create PyInstaller spec file
    spec_content = create_spec_file()
    
    # Write spec file
    with open("cybersecurity_engine.spec", "w") as f:
        f.write(spec_content)
    
    # Create executable
    print("🔨 Building executable...")
    try:
        result = subprocess.run([
            "pyinstaller", 
            "--onefile",
            "--windowed",
            "--name=CybersecurityEngine",
            "--icon=icon.ico" if os.path.exists("icon.ico") else "",
            "cybersecurity_launcher.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Executable created successfully!")
            print("📁 Location: dist/CybersecurityEngine.exe")
            return True
        else:
            print(f"❌ Failed to create executable: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating executable: {e}")
        return False

def create_launcher_script():
    """Create the main launcher script"""
    launcher_content = '''#!/usr/bin/env python3
"""
Self-Morphing AI Cybersecurity Engine - Launcher
Professional Cybersecurity Platform
"""

import sys
import os
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import requests
import json

class CybersecurityEngineGUI:
    """GUI for Self-Morphing AI Cybersecurity Engine"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🛡️ Self-Morphing AI Cybersecurity Engine")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a1a')
        
        # Engine processes
        self.api_process = None
        self.dashboard_process = None
        self.engine_process = None
        
        # Status variables
        self.api_running = False
        self.dashboard_running = False
        self.engine_running = False
        
        self.setup_gui()
        self.check_system()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#1a1a1a')
        title_frame.pack(fill='x', padx=20, pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="🛡️ Self-Morphing AI Cybersecurity Engine",
            font=('Arial', 16, 'bold'),
            fg='#00ff00',
            bg='#1a1a1a'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Professional Cybersecurity Platform",
            font=('Arial', 12),
            fg='#cccccc',
            bg='#1a1a1a'
        )
        subtitle_label.pack()
        
        # Status frame
        status_frame = tk.LabelFrame(
            self.root, 
            text="System Status", 
            font=('Arial', 12, 'bold'),
            fg='#00ff00',
            bg='#1a1a1a',
            relief='groove'
        )
        status_frame.pack(fill='x', padx=20, pady=10)
        
        # Status indicators
        self.api_status = tk.Label(
            status_frame, 
            text="API Server: ❌ Stopped", 
            font=('Arial', 10),
            fg='#ff4444',
            bg='#1a1a1a'
        )
        self.api_status.pack(anchor='w', padx=10, pady=5)
        
        self.dashboard_status = tk.Label(
            status_frame, 
            text="Dashboard: ❌ Stopped", 
            font=('Arial', 10),
            fg='#ff4444',
            bg='#1a1a1a'
        )
        self.dashboard_status.pack(anchor='w', padx=10, pady=5)
        
        self.engine_status = tk.Label(
            status_frame, 
            text="AI Engine: ❌ Stopped", 
            font=('Arial', 10),
            fg='#ff4444',
            bg='#1a1a1a'
        )
        self.engine_status.pack(anchor='w', padx=10, pady=5)
        
        # Control buttons
        control_frame = tk.Frame(self.root, bg='#1a1a1a')
        control_frame.pack(fill='x', padx=20, pady=10)
        
        self.start_button = tk.Button(
            control_frame,
            text="🚀 Start Cybersecurity Engine",
            font=('Arial', 12, 'bold'),
            bg='#00aa00',
            fg='white',
            command=self.start_engine,
            width=25
        )
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = tk.Button(
            control_frame,
            text="⏹️ Stop Engine",
            font=('Arial', 12, 'bold'),
            bg='#aa0000',
            fg='white',
            command=self.stop_engine,
            width=15,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=5)
        
        self.training_button = tk.Button(
            control_frame,
            text="🧠 Train AI Models",
            font=('Arial', 12, 'bold'),
            bg='#0066cc',
            fg='white',
            command=self.start_training,
            width=20
        )
        self.training_button.pack(side='left', padx=5)
        
        # Dashboard buttons
        dashboard_frame = tk.Frame(self.root, bg='#1a1a1a')
        dashboard_frame.pack(fill='x', padx=20, pady=10)
        
        self.dashboard_button = tk.Button(
            dashboard_frame,
            text="📊 Open Security Dashboard",
            font=('Arial', 12, 'bold'),
            bg='#0066cc',
            fg='white',
            command=self.open_dashboard,
            width=25,
            state='disabled'
        )
        self.dashboard_button.pack(side='left', padx=5)
        
        self.api_button = tk.Button(
            dashboard_frame,
            text="🔧 Open API Documentation",
            font=('Arial', 12, 'bold'),
            bg='#0066cc',
            fg='white',
            command=self.open_api_docs,
            width=25,
            state='disabled'
        )
        self.api_button.pack(side='left', padx=5)
        
        # Log output
        log_frame = tk.LabelFrame(
            self.root, 
            text="System Logs", 
            font=('Arial', 12, 'bold'),
            fg='#00ff00',
            bg='#1a1a1a',
            relief='groove'
        )
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=15,
            font=('Consolas', 9),
            bg='#000000',
            fg='#00ff00',
            insertbackground='#00ff00'
        )
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root, 
            mode='indeterminate',
            length=400
        )
        self.progress.pack(pady=10)
        
        # Start status monitoring
        self.update_status()
    
    def log_message(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def check_system(self):
        """Check system requirements"""
        self.log_message("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.log_message("❌ Python 3.8+ required")
            return False
        
        # Check required files
        required_files = [
            "backend/api_server.py",
            "backend/main_engine.py",
            "backend/order_engine.py",
            "backend/chaos_engine.py"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            self.log_message(f"❌ Missing files: {', '.join(missing_files)}")
            return False
        
        self.log_message("✅ System requirements met")
        return True
    
    def start_engine(self):
        """Start the cybersecurity engine"""
        self.log_message("🚀 Starting Self-Morphing AI Cybersecurity Engine...")
        self.progress.start()
        
        try:
            # Start API server
            self.log_message("📡 Starting API server...")
            self.api_process = subprocess.Popen([
                sys.executable, "backend/api_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for API server to start
            time.sleep(5)
            
            # Check if API is running
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    self.api_running = True
                    self.log_message("✅ API server started successfully")
                else:
                    self.log_message("❌ API server failed to start")
                    return
            except:
                self.log_message("❌ API server not responding")
                return
            
            # Start main engine
            self.log_message("🧠 Starting AI engine...")
            self.engine_process = subprocess.Popen([
                sys.executable, "backend/main_engine.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Start dashboard
            self.log_message("📊 Starting security dashboard...")
            self.dashboard_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "backend/dashboard.py",
                "--server.port=8501", "--server.headless=true"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for dashboard to start
            time.sleep(10)
            
            self.engine_running = True
            self.dashboard_running = True
            
            self.log_message("✅ Cybersecurity engine started successfully!")
            self.log_message("🛡️ System ready for cybersecurity operations")
            
            # Update button states
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.dashboard_button.config(state='normal')
            self.api_button.config(state='normal')
            
            self.progress.stop()
            
        except Exception as e:
            self.log_message(f"❌ Failed to start engine: {e}")
            self.progress.stop()
    
    def stop_engine(self):
        """Stop the cybersecurity engine"""
        self.log_message("⏹️ Stopping cybersecurity engine...")
        
        try:
            # Stop processes
            if self.api_process:
                self.api_process.terminate()
                self.api_running = False
                self.log_message("✅ API server stopped")
            
            if self.engine_process:
                self.engine_process.terminate()
                self.engine_running = False
                self.log_message("✅ AI engine stopped")
            
            if self.dashboard_process:
                self.dashboard_process.terminate()
                self.dashboard_running = False
                self.log_message("✅ Dashboard stopped")
            
            self.log_message("✅ Cybersecurity engine stopped")
            
            # Update button states
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.dashboard_button.config(state='disabled')
            self.api_button.config(state='disabled')
            
        except Exception as e:
            self.log_message(f"❌ Error stopping engine: {e}")
    
    def start_training(self):
        """Start AI model training"""
        self.log_message("🧠 Starting AI model training...")
        
        def run_training():
            try:
                # Run enhanced training
                result = subprocess.run([
                    sys.executable, "backend/train_from_scratch.py"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_message("✅ AI model training completed successfully!")
                else:
                    self.log_message(f"❌ Training failed: {result.stderr}")
                    
            except Exception as e:
                self.log_message(f"❌ Training error: {e}")
        
        # Run training in separate thread
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
    
    def open_dashboard(self):
        """Open security dashboard"""
        try:
            webbrowser.open("http://localhost:8501")
            self.log_message("📊 Opening security dashboard...")
        except Exception as e:
            self.log_message(f"❌ Failed to open dashboard: {e}")
    
    def open_api_docs(self):
        """Open API documentation"""
        try:
            webbrowser.open("http://localhost:8000/docs")
            self.log_message("🔧 Opening API documentation...")
        except Exception as e:
            self.log_message(f"❌ Failed to open API docs: {e}")
    
    def update_status(self):
        """Update system status"""
        # Update API status
        if self.api_running:
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    self.api_status.config(text="API Server: ✅ Running", fg='#00ff00')
                else:
                    self.api_status.config(text="API Server: ⚠️ Error", fg='#ffaa00')
            except:
                self.api_status.config(text="API Server: ❌ Stopped", fg='#ff4444')
                self.api_running = False
        
        # Update dashboard status
        if self.dashboard_running:
            try:
                response = requests.get("http://localhost:8501", timeout=2)
                if response.status_code == 200:
                    self.dashboard_status.config(text="Dashboard: ✅ Running", fg='#00ff00')
                else:
                    self.dashboard_status.config(text="Dashboard: ⚠️ Error", fg='#ffaa00')
            except:
                self.dashboard_status.config(text="Dashboard: ❌ Stopped", fg='#ff4444')
                self.dashboard_running = False
        
        # Update engine status
        if self.engine_running:
            self.engine_status.config(text="AI Engine: ✅ Running", fg='#00ff00')
        else:
            self.engine_status.config(text="AI Engine: ❌ Stopped", fg='#ff4444')
        
        # Schedule next update
        self.root.after(5000, self.update_status)
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def main():
    """Main function"""
    print("🛡️ Self-Morphing AI Cybersecurity Engine")
    print("Professional Cybersecurity Platform")
    print("=" * 50)
    
    # Create and run GUI
    app = CybersecurityEngineGUI()
    app.run()

if __name__ == "__main__":
    main()
'''
    
    # Write launcher script
    with open("cybersecurity_launcher.py", "w") as f:
        f.write(launcher_content)
    
    print("✅ Launcher script created")
    return launcher_content

def create_spec_file():
    """Create PyInstaller spec file"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['cybersecurity_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('backend', 'backend'),
        ('frontend', 'frontend'),
        ('requirements.txt', '.'),
        ('README.md', '.'),
        ('*.py', '.'),
    ],
    hiddenimports=[
        'fastapi',
        'uvicorn',
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'requests',
        'tkinter',
        'threading',
        'subprocess',
        'webbrowser'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CybersecurityEngine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None
)
'''
    return spec_content

def create_requirements():
    """Create requirements.txt for executable"""
    requirements = '''fastapi>=0.104.1
uvicorn>=0.24.0
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
requests>=2.31.0
tkinter
pyinstaller>=5.0
'''
    
    with open("requirements_executable.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Requirements file created")

def create_build_script():
    """Create build script"""
    build_script = '''#!/usr/bin/env python3
"""
Build script for Self-Morphing AI Cybersecurity Engine
"""

import os
import sys
import subprocess
import shutil

def build_executable():
    """Build the executable"""
    print("🛡️ Building Self-Morphing AI Cybersecurity Engine Executable")
    print("=" * 60)
    
    # Install requirements
    print("📦 Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_executable.txt"])
    
    # Create executable
    print("🔨 Creating executable...")
    subprocess.run([
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=CybersecurityEngine",
        "--add-data=backend;backend",
        "--add-data=frontend;frontend",
        "--add-data=requirements.txt;.",
        "--add-data=README.md;.",
        "--hidden-import=fastapi",
        "--hidden-import=uvicorn",
        "--hidden-import=streamlit",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=scikit-learn",
        "--hidden-import=requests",
        "--hidden-import=tkinter",
        "cybersecurity_launcher.py"
    ])
    
    print("✅ Executable created successfully!")
    print("📁 Location: dist/CybersecurityEngine.exe")

if __name__ == "__main__":
    build_executable()
'''
    
    with open("build_executable.py", "w") as f:
        f.write(build_script)
    
    print("✅ Build script created")

def main():
    """Main function"""
    print("🛡️ Self-Morphing AI Cybersecurity Engine - Executable Creator")
    print("=" * 70)
    
    # Create all necessary files
    create_launcher_script()
    create_requirements()
    create_build_script()
    
    print("\\n📁 Files created:")
    print("  - cybersecurity_launcher.py (Main launcher)")
    print("  - requirements_executable.txt (Dependencies)")
    print("  - build_executable.py (Build script)")
    
    print("\\n🚀 To create executable:")
    print("  1. Install PyInstaller: pip install pyinstaller")
    print("  2. Run build script: python build_executable.py")
    print("  3. Find executable: dist/CybersecurityEngine.exe")
    
    print("\\n✅ Executable creation setup complete!")

if __name__ == "__main__":
    main()
