#!/usr/bin/env python3
"""
Autonomous Deployment Script for Self-Morphing AI Cybersecurity Engine
Deploys the system for background operation without human guidance
"""

import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path

class AutonomousDeployment:
    """Deploy the cybersecurity engine for autonomous operation"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.project_dir = Path(__file__).parent
        self.deployment_dir = self.project_dir / "deployment"
        
    def check_requirements(self):
        """Check system requirements for autonomous deployment"""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher required")
            return False
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.total < 2 * 1024 * 1024 * 1024:  # 2GB
                print("⚠️  Warning: Less than 2GB RAM available, performance may be affected")
        except ImportError:
            print("⚠️  psutil not available, cannot check memory")
        
        # Check disk space
        disk = shutil.disk_usage('.')
        if disk.free < 1 * 1024 * 1024 * 1024:  # 1GB
            print("❌ Less than 1GB disk space available")
            return False
        
        print("✅ System requirements check passed")
        return True
    
    def create_directories(self):
        """Create required directories for autonomous operation"""
        print("📁 Creating directories...")
        
        directories = [
            'data', 'models', 'logs', 'config', 'monitoring',
            'deployment', 'backups'
        ]
        
        for directory in directories:
            dir_path = self.project_dir / directory
            dir_path.mkdir(exist_ok=True)
            print(f"  ✅ Created: {directory}")
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("📦 Installing dependencies...")
        
        try:
            # Install Python dependencies
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 
                str(self.project_dir / 'backend' / 'requirements.txt')
            ], check=True)
            
            # Install additional dependencies for autonomous operation
            additional_deps = [
                'psutil', 'requests', 'schedule', 'watchdog'
            ]
            
            for dep in additional_deps:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', dep
                ], check=True)
            
            print("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_autonomous_config(self):
        """Setup configuration for autonomous operation"""
        print("⚙️  Setting up autonomous configuration...")
        
        config_file = self.project_dir / 'config' / 'autonomous_config.json'
        
        if not config_file.exists():
            print("❌ Autonomous configuration file not found")
            return False
        
        # Load and validate configuration
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print("✅ Autonomous configuration loaded")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            return False
    
    def setup_platform_specific(self):
        """Setup platform-specific configurations"""
        print(f"🖥️  Setting up for {self.platform}...")
        
        if self.platform == 'windows':
            return self._setup_windows()
        elif self.platform == 'linux':
            return self._setup_linux()
        elif self.platform == 'darwin':
            return self._setup_macos()
        else:
            print(f"⚠️  Unsupported platform: {self.platform}")
            return False
    
    def _setup_windows(self):
        """Setup Windows-specific configurations"""
        print("  Setting up Windows service...")
        
        # Check if pywin32 is available
        try:
            import win32serviceutil
            print("  ✅ Windows service support available")
            return True
        except ImportError:
            print("  ⚠️  pywin32 not available, Windows service not supported")
            print("  💡 Install with: pip install pywin32")
            return False
    
    def _setup_linux(self):
        """Setup Linux-specific configurations"""
        print("  Setting up Linux systemd service...")
        
        # Check if running as root
        if os.geteuid() != 0:
            print("  ⚠️  Root privileges required for systemd service setup")
            print("  💡 Run with: sudo python deploy_autonomous.py")
            return False
        
        # Copy systemd service file
        service_file = self.project_dir / 'cybersecurity-engine.service'
        if service_file.exists():
            try:
                shutil.copy2(service_file, '/etc/systemd/system/')
                subprocess.run(['systemctl', 'daemon-reload'], check=True)
                print("  ✅ Systemd service file installed")
                return True
            except Exception as e:
                print(f"  ❌ Failed to install systemd service: {e}")
                return False
        
        return True
    
    def _setup_macos(self):
        """Setup macOS-specific configurations"""
        print("  Setting up macOS launchd service...")
        
        # Create launchd plist file
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.cybersecurity.engine</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{self.project_dir}/autonomous_start.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{self.project_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{self.project_dir}/logs/service.log</string>
    <key>StandardErrorPath</key>
    <string>{self.project_dir}/logs/service.error.log</string>
</dict>
</plist>"""
        
        plist_file = self.project_dir / 'com.cybersecurity.engine.plist'
        with open(plist_file, 'w') as f:
            f.write(plist_content)
        
        print("  ✅ Launchd plist file created")
        print(f"  💡 Install with: sudo cp {plist_file} /Library/LaunchDaemons/")
        print("  💡 Load with: sudo launchctl load /Library/LaunchDaemons/com.cybersecurity.engine.plist")
        
        return True
    
    def create_startup_scripts(self):
        """Create startup scripts for different platforms"""
        print("📜 Creating startup scripts...")
        
        # Windows batch file
        windows_script = self.project_dir / 'start_autonomous.bat'
        with open(windows_script, 'w') as f:
            f.write(f"""@echo off
echo Starting Autonomous Cybersecurity Engine...
cd /d "{self.project_dir}"
python autonomous_start.py
pause
""")
        
        # Linux/Mac shell script
        unix_script = self.project_dir / 'start_autonomous.sh'
        with open(unix_script, 'w') as f:
            f.write(f"""#!/bin/bash
echo "Starting Autonomous Cybersecurity Engine..."
cd "{self.project_dir}"
python3 autonomous_start.py
""")
        
        # Make shell script executable
        if self.platform != 'windows':
            os.chmod(unix_script, 0o755)
        
        print("  ✅ Startup scripts created")
        return True
    
    def test_deployment(self):
        """Test the deployment"""
        print("🧪 Testing deployment...")
        
        try:
            # Test import of main modules
            sys.path.insert(0, str(self.project_dir / 'backend'))
            
            import main_engine
            import order_engine
            import chaos_engine
            import balance_controller
            
            print("  ✅ Core modules import successfully")
            
            # Test configuration loading
            config_file = self.project_dir / 'config' / 'autonomous_config.json'
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print("  ✅ Configuration loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Deployment test failed: {e}")
            return False
    
    def deploy(self):
        """Main deployment function"""
        print("🚀 Deploying Autonomous Self-Morphing AI Cybersecurity Engine...")
        print("=" * 60)
        
        # Check requirements
        if not self.check_requirements():
            return False
        
        # Create directories
        self.create_directories()
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Setup configuration
        if not self.setup_autonomous_config():
            return False
        
        # Setup platform-specific configurations
        if not self.setup_platform_specific():
            print("⚠️  Platform-specific setup failed, but deployment can continue")
        
        # Create startup scripts
        self.create_startup_scripts()
        
        # Test deployment
        if not self.test_deployment():
            return False
        
        print("\n🎉 Deployment completed successfully!")
        print("\n📋 Next Steps:")
        
        if self.platform == 'windows':
            print("  • Run: python autonomous_start.py")
            print("  • Or: start_autonomous.bat")
            print("  • For service: python install_windows_service.py install")
        elif self.platform == 'linux':
            print("  • Run: python3 autonomous_start.py")
            print("  • Or: ./start_autonomous.sh")
            print("  • For service: sudo systemctl enable cybersecurity-engine")
        elif self.platform == 'darwin':
            print("  • Run: python3 autonomous_start.py")
            print("  • Or: ./start_autonomous.sh")
            print("  • For service: sudo launchctl load /Library/LaunchDaemons/com.cybersecurity.engine.plist")
        
        print("\n🔧 Configuration: config/autonomous_config.json")
        print("📊 Dashboard: http://localhost:8501")
        print("🔗 API: http://localhost:8000")
        print("📋 Health: http://localhost:8000/health")
        
        return True

def main():
    """Main deployment function"""
    deployment = AutonomousDeployment()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Autonomous Deployment Script")
        print("Usage: python deploy_autonomous.py")
        print("\nThis script deploys the cybersecurity engine for autonomous operation.")
        return
    
    success = deployment.deploy()
    
    if success:
        print("\n✅ Deployment successful!")
        sys.exit(0)
    else:
        print("\n❌ Deployment failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()


