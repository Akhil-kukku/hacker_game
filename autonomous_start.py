#!/usr/bin/env python3
"""
Autonomous Self-Morphing AI Cybersecurity Engine
Runs the complete system in background mode without human guidance
"""

import subprocess
import sys
import os
import time
import signal
import threading
import logging
import json
from pathlib import Path
from datetime import datetime
import requests
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - AUTONOMOUS - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous.log'),
        logging.StreamHandler()
    ]
)

class AutonomousCybersecurityEngine:
    """Autonomous cybersecurity engine that runs without human guidance"""
    
    def __init__(self):
        self.running = False
        self.api_process = None
        self.dashboard_process = None
        self.monitoring_thread = None
        self.health_check_interval = 30  # seconds
        self.max_restart_attempts = 5
        self.restart_delay = 10  # seconds
        
        # Performance monitoring
        self.performance_metrics = {
            'start_time': datetime.now(),
            'restart_count': 0,
            'uptime': 0,
            'last_health_check': None,
            'system_load': 0.0,
            'memory_usage': 0.0
        }
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def check_system_requirements(self):
        """Check if system meets requirements for autonomous operation"""
        try:
            # Check available memory (minimum 2GB)
            memory = psutil.virtual_memory()
            if memory.total < 2 * 1024 * 1024 * 1024:  # 2GB
                logging.warning("System has less than 2GB RAM, performance may be affected")
            
            # Check available disk space (minimum 1GB)
            disk = psutil.disk_usage('.')
            if disk.free < 1 * 1024 * 1024 * 1024:  # 1GB
                logging.warning("Low disk space, may affect operation")
            
            # Check Python version
            if sys.version_info < (3, 8):
                raise Exception("Python 3.8 or higher required")
            
            logging.info("System requirements check passed")
            return True
            
        except Exception as e:
            logging.error(f"System requirements check failed: {e}")
            return False
    
    def create_directories(self):
        """Create required directories for autonomous operation"""
        directories = ['data', 'models', 'logs', 'config', 'monitoring']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logging.info(f"Created directory: {directory}")
    
    def start_api_server(self):
        """Start the API server in background"""
        try:
            logging.info("Starting API server...")
            self.api_process = subprocess.Popen([
                sys.executable, 'backend/api_server.py'
            ], cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for API to be ready
            if self._wait_for_api_ready():
                logging.info("API server started successfully")
                return True
            else:
                logging.error("API server failed to start")
                return False
                
        except Exception as e:
            logging.error(f"Failed to start API server: {e}")
            return False
    
    def start_dashboard(self):
        """Start the dashboard (optional for monitoring)"""
        try:
            logging.info("Starting dashboard...")
            self.dashboard_process = subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run', 'backend/dashboard.py',
                '--server.port', '8501',
                '--server.headless', 'true',
                '--server.enableCORS', 'false',
                '--server.enableXsrfProtection', 'false'
            ], cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logging.info("Dashboard started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start dashboard: {e}")
            return False
    
    def _wait_for_api_ready(self, max_attempts=30):
        """Wait for API server to be ready"""
        for attempt in range(max_attempts):
            try:
                response = requests.get('http://localhost:8000/health', timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
            if attempt % 5 == 0:
                logging.info(f"Waiting for API server... ({attempt + 1}/{max_attempts})")
        return False
    
    def start_monitoring(self):
        """Start system monitoring thread"""
        def monitor_system():
            while self.running:
                try:
                    # Check API health
                    if not self._check_api_health():
                        logging.warning("API health check failed, attempting restart...")
                        self._restart_api_server()
                    
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    # Log system status
                    self._log_system_status()
                    
                    time.sleep(self.health_check_interval)
                    
                except Exception as e:
                    logging.error(f"Monitoring error: {e}")
                    time.sleep(5)
        
        self.monitoring_thread = threading.Thread(target=monitor_system, daemon=True)
        self.monitoring_thread.start()
        logging.info("System monitoring started")
    
    def _check_api_health(self):
        """Check if API server is healthy"""
        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _restart_api_server(self):
        """Restart API server if it fails"""
        try:
            if self.api_process:
                self.api_process.terminate()
                self.api_process.wait(timeout=10)
            
            time.sleep(self.restart_delay)
            self.start_api_server()
            self.performance_metrics['restart_count'] += 1
            logging.info(f"API server restarted (attempt {self.performance_metrics['restart_count']})")
            
        except Exception as e:
            logging.error(f"Failed to restart API server: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Update uptime
            self.performance_metrics['uptime'] = (datetime.now() - self.performance_metrics['start_time']).total_seconds()
            
            # Update system load
            self.performance_metrics['system_load'] = psutil.cpu_percent()
            
            # Update memory usage
            memory = psutil.virtual_memory()
            self.performance_metrics['memory_usage'] = memory.percent
            
            # Update last health check
            self.performance_metrics['last_health_check'] = datetime.now().isoformat()
            
        except Exception as e:
            logging.error(f"Failed to update performance metrics: {e}")
    
    def _log_system_status(self):
        """Log system status periodically"""
        try:
            # Get system status from API
            response = requests.get('http://localhost:8000/status', timeout=5)
            if response.status_code == 200:
                status = response.json()
                
                # Log key metrics
                logging.info(f"System Status - Balance: {status.get('performance_metrics', {}).get('system_balance_score', 0.0):.3f}, "
                           f"Simulations: {status.get('performance_metrics', {}).get('total_simulations', 0)}, "
                           f"Uptime: {self.performance_metrics['uptime']:.0f}s, "
                           f"Load: {self.performance_metrics['system_load']:.1f}%, "
                           f"Memory: {self.performance_metrics['memory_usage']:.1f}%")
                
        except Exception as e:
            logging.error(f"Failed to log system status: {e}")
    
    def save_system_state(self):
        """Save system state for recovery"""
        try:
            state_data = {
                'performance_metrics': self.performance_metrics,
                'timestamp': datetime.now().isoformat(),
                'restart_count': self.performance_metrics['restart_count'],
                'uptime': self.performance_metrics['uptime']
            }
            
            with open('data/autonomous_state.json', 'w') as f:
                json.dump(state_data, f, indent=2)
                
            logging.info("System state saved")
            
        except Exception as e:
            logging.error(f"Failed to save system state: {e}")
    
    def load_system_state(self):
        """Load previous system state"""
        try:
            state_file = 'data/autonomous_state.json'
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Restore performance metrics
                self.performance_metrics.update(state_data.get('performance_metrics', {}))
                logging.info("System state loaded successfully")
                
        except Exception as e:
            logging.error(f"Failed to load system state: {e}")
    
    def start(self):
        """Start the autonomous cybersecurity engine"""
        try:
            logging.info("🛡️ Starting Autonomous Self-Morphing AI Cybersecurity Engine...")
            
            # Check system requirements
            if not self.check_system_requirements():
                raise Exception("System requirements not met")
            
            # Create directories
            self.create_directories()
            
            # Load previous state
            self.load_system_state()
            
            # Start API server
            if not self.start_api_server():
                raise Exception("Failed to start API server")
            
            # Start dashboard (optional)
            self.start_dashboard()
            
            # Start monitoring
            self.start_monitoring()
            
            self.running = True
            
            logging.info("✅ Autonomous cybersecurity engine started successfully")
            logging.info("📊 Dashboard: http://localhost:8501")
            logging.info("🔗 API: http://localhost:8000")
            logging.info("📋 Health: http://localhost:8000/health")
            logging.info("🤖 Running in autonomous mode - no human guidance required")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start autonomous engine: {e}")
            return False
    
    def run(self):
        """Main run loop"""
        if not self.start():
            return False
        
        try:
            # Save state every 5 minutes
            last_save = time.time()
            save_interval = 300  # 5 minutes
            
            while self.running:
                time.sleep(1)
                
                # Check if processes are still running
                if self.api_process and self.api_process.poll() is not None:
                    logging.error("API server stopped unexpectedly")
                    if self.performance_metrics['restart_count'] < self.max_restart_attempts:
                        self._restart_api_server()
                    else:
                        logging.error("Maximum restart attempts reached, shutting down")
                        break
                
                # Save state periodically
                if time.time() - last_save > save_interval:
                    self.save_system_state()
                    last_save = time.time()
                
        except KeyboardInterrupt:
            logging.info("Received interrupt signal")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the autonomous engine"""
        try:
            logging.info("🛑 Shutting down autonomous cybersecurity engine...")
            
            self.running = False
            
            # Save final state
            self.save_system_state()
            
            # Terminate processes
            if self.api_process:
                self.api_process.terminate()
                self.api_process.wait(timeout=10)
            
            if self.dashboard_process:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=10)
            
            logging.info("✅ Autonomous cybersecurity engine shutdown complete")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

def main():
    """Main entry point for autonomous operation"""
    engine = AutonomousCybersecurityEngine()
    
    try:
        engine.run()
    except Exception as e:
        logging.error(f"Autonomous engine failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
