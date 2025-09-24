#!/usr/bin/env python3
"""
Windows Service Installer for Self-Morphing AI Cybersecurity Engine
Installs the engine as a Windows service for background operation
"""

import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import time
import sys
import os
import logging
import subprocess
import threading
from pathlib import Path

class CybersecurityEngineService(win32serviceutil.ServiceFramework):
    """Windows service for the cybersecurity engine"""
    
    _svc_name_ = "CybersecurityEngine"
    _svc_display_name_ = "Self-Morphing AI Cybersecurity Engine"
    _svc_description_ = "Autonomous AI-powered cybersecurity defense system"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_running = True
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - SERVICE - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('C:\\ProgramData\\CybersecurityEngine\\service.log'),
                logging.StreamHandler()
            ]
        )
    
    def SvcStop(self):
        """Stop the service"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_running = False
        logging.info("Service stop requested")
    
    def SvcDoRun(self):
        """Main service execution"""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        logging.info("Cybersecurity Engine service started")
        
        try:
            # Start the autonomous engine
            self._start_autonomous_engine()
            
            # Wait for stop signal
            while self.is_running:
                win32event.WaitForSingleObject(self.hWaitStop, 1000)
                
        except Exception as e:
            logging.error(f"Service error: {e}")
            servicemanager.LogErrorMsg(f"Service error: {e}")
    
    def _start_autonomous_engine(self):
        """Start the autonomous cybersecurity engine"""
        try:
            # Change to the application directory
            app_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(app_dir)
            
            # Start the autonomous engine
            self.engine_process = subprocess.Popen([
                sys.executable, 'autonomous_start.py'
            ], cwd=app_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logging.info("Autonomous cybersecurity engine started")
            
            # Monitor the engine process
            while self.is_running and self.engine_process.poll() is None:
                time.sleep(1)
            
            if self.engine_process.poll() is not None:
                logging.error("Autonomous engine process stopped unexpectedly")
                self.engine_process.terminate()
                
        except Exception as e:
            logging.error(f"Failed to start autonomous engine: {e}")
            raise

def install_service():
    """Install the Windows service"""
    try:
        # Create service user account (if needed)
        # This would typically be done by an administrator
        
        # Install the service
        win32serviceutil.InstallService(
            CybersecurityEngineService._svc_name_,
            CybersecurityEngineService._svc_display_name_,
            CybersecurityEngineService._svc_description_,
            startType=win32service.SERVICE_AUTO_START
        )
        
        print("✅ Cybersecurity Engine service installed successfully")
        print("🔧 Use 'net start CybersecurityEngine' to start the service")
        print("🔧 Use 'net stop CybersecurityEngine' to stop the service")
        
    except Exception as e:
        print(f"❌ Failed to install service: {e}")

def uninstall_service():
    """Uninstall the Windows service"""
    try:
        win32serviceutil.RemoveService(CybersecurityEngineService._svc_name_)
        print("✅ Cybersecurity Engine service uninstalled successfully")
        
    except Exception as e:
        print(f"❌ Failed to uninstall service: {e}")

def main():
    """Main entry point"""
    if len(sys.argv) == 1:
        # Run as service
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(CybersecurityEngineService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        # Handle command line arguments
        if sys.argv[1] == 'install':
            install_service()
        elif sys.argv[1] == 'uninstall':
            uninstall_service()
        elif sys.argv[1] == 'start':
            win32serviceutil.StartService(CybersecurityEngineService._svc_name_)
            print("✅ Service started")
        elif sys.argv[1] == 'stop':
            win32serviceutil.StopService(CybersecurityEngineService._svc_name_)
            print("✅ Service stopped")
        elif sys.argv[1] == 'restart':
            win32serviceutil.RestartService(CybersecurityEngineService._svc_name_)
            print("✅ Service restarted")
        else:
            print("Usage: python install_windows_service.py [install|uninstall|start|stop|restart]")

if __name__ == '__main__':
    main()
