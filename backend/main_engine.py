"""
Self-Morphing AI Cybersecurity Engine - Main Orchestrator
Coordinates ORDER (Defense), CHAOS (Offense), and BALANCE (Controller) components
"""

import asyncio
import threading
import time
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import signal
import sys
import os

from order_engine import OrderEngine, NetworkFlow
from chaos_engine import ChaosEngine, AttackType
from balance_controller import BalanceController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MAIN - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main_engine.log'),
        logging.StreamHandler()
    ]
)

class SelfMorphingAICybersecurityEngine:
    """
    Main orchestrator for the Self-Morphing AI Cybersecurity Engine
    Coordinates ORDER, CHAOS, and BALANCE components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.order_engine = None
        self.chaos_engine = None
        self.balance_controller = None
        
        # System state
        self.running = False
        self.simulation_mode = True
        self.batch_simulations = []
        self.attack_response_tracking = []
        
        # NEW: Flow and attack correlation tracking
        self.flow_cache = {}  # flow_id -> NetworkFlow object
        self.flow_attack_map = {}  # flow_id -> attack_id mapping
        self.attack_target_map = {}  # attack_id -> [flow_ids] that were targeted
        self.detected_anomalies = set()  # Set of flow_ids detected as anomalies
        
        # Performance tracking
        self.performance_metrics = {
            'total_simulations': 0,
            'successful_defenses': 0,
            'successful_attacks': 0,
            'system_balance_score': 0.0,
            'last_simulation': None,
            'total_runtime': 0.0,
            'true_positives': 0,  # Attacks correctly detected
            'false_negatives': 0,  # Attacks missed
            'false_positives': 0,  # Normal traffic flagged as attack
            'true_negatives': 0   # Normal traffic correctly passed
        }
        # Feedback toggles
        self.enable_online_feedback = True
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logging.info("Self-Morphing AI Cybersecurity Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the main engine"""
        return {
            'simulation_interval': 5.0,  # seconds
            'batch_size': 100,
            'max_simulations': 1000,
            'performance_threshold': 0.7,
            'auto_optimization': True,
            'save_interval': 300,  # 5 minutes
            'log_level': 'INFO',
            'simulation_mode': True,
            'real_time_mode': False,
            'data_dir': 'data',
            'models_dir': 'models',
            'logs_dir': 'logs'
        }
    
    def _initialize_components(self):
        """Initialize all three components"""
        try:
            # Create directories
            os.makedirs(self.config['data_dir'], exist_ok=True)
            os.makedirs(self.config['models_dir'], exist_ok=True)
            os.makedirs(self.config['logs_dir'], exist_ok=True)
            
            # Initialize ORDER Engine (Defense)
            order_config = {
                'contamination': 0.1,
                'n_estimators': 100,
                'max_samples': 'auto',
                'random_state': 42,
                'batch_size': 500,
                'training_threshold': 200,
                'mutation_threshold': 0.8,
                'max_signatures': 1000,
                'confidence_threshold': 0.7,
                'model_save_path': f"{self.config['models_dir']}/order_model.pkl",
                'scaler_save_path': f"{self.config['models_dir']}/order_scaler.pkl",
                'signatures_save_path': f"{self.config['data_dir']}/attack_signatures.json"
            }
            self.order_engine = OrderEngine(order_config)
            # Load saved ORDER model if present
            try:
                self.order_engine.load_model()
            except Exception as e:
                logging.warning(f"ORDER model not loaded: {e}")
            
            # Initialize CHAOS Engine (Offense)
            chaos_config = {
                'max_concurrent_attacks': 5,
                'attack_interval': 1.0,
                'stealth_threshold': 0.7,
                'adaptation_threshold': 0.3,
                'max_attack_history': 1000,
                'payload_size_range': (64, 4096),
                'timeout': 30.0,
                'retry_attempts': 3,
                'evolution_rate': 0.1,
                'mutation_probability': 0.2
            }
            self.chaos_engine = ChaosEngine(chaos_config)
            
            # Initialize BALANCE Controller
            balance_config = {
                'experience_buffer_size': 1000,
                'population_size': 50,
                'control_interval': 5.0,
                'initial_epsilon': 0.1,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'elite_size': 5,
                'generation_limit': 100,
                'fitness_threshold': 0.8,
                'optimization_threshold': 0.7,
                'save_path': f"{self.config['models_dir']}/balance_controller.pkl"
            }
            self.balance_controller = BalanceController(balance_config)
            
            logging.info("All components initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """Start the main engine"""
        try:
            logging.info("Starting Self-Morphing AI Cybersecurity Engine...")
            
            self.running = True
            
            # Start BALANCE controller
            self.balance_controller.start_control_loop()
            
            # Start main simulation loop
            if self.simulation_mode:
                self._start_simulation_loop()
            else:
                self._start_real_time_loop()
            
            logging.info("Engine started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start engine: {e}")
            raise
    
    def _start_simulation_loop(self):
        """Start the simulation loop"""
        def simulation_thread():
            while self.running:
                try:
                    # Run batch simulation
                    self._run_batch_simulation()
                    
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    # Check if optimization is needed
                    if self.config['auto_optimization'] and self._should_optimize():
                        self._optimize_system()
                    
                    # Sleep for simulation interval
                    time.sleep(self.config['simulation_interval'])
                    
                except Exception as e:
                    logging.error(f"Error in simulation loop: {e}")
                    time.sleep(1)
        
        # Start simulation thread
        sim_thread = threading.Thread(target=simulation_thread, daemon=True)
        sim_thread.start()
        logging.info("Simulation loop started")
    
    def _start_real_time_loop(self):
        """Start the real-time processing loop"""
        def real_time_thread():
            while self.running:
                try:
                    # Process real-time network flows
                    self._process_real_time_flows()
                    
                    # Process real-time attacks
                    self._process_real_time_attacks()
                    
                    # Update system state
                    self._update_system_state()
                    
                    time.sleep(1)  # 1 second intervals
                    
                except Exception as e:
                    logging.error(f"Error in real-time loop: {e}")
                    time.sleep(1)
        
        # Start real-time thread
        rt_thread = threading.Thread(target=real_time_thread, daemon=True)
        rt_thread.start()
        logging.info("Real-time loop started")
    
    def _run_batch_simulation(self):
        """Run a batch simulation"""
        try:
            logging.info("Starting batch simulation")
            
            # Generate simulated network flows
            flows = self._generate_simulated_flows()
            
            # Generate simulated attacks
            attacks = self._generate_simulated_attacks()
            
            # Process flows through ORDER engine
            defense_results = self._process_flows_through_order(flows)
            
            # Process attacks through CHAOS engine
            attack_results = self._process_attacks_through_chaos(attacks)
            
            # Track attack-response interactions
            interactions = self._track_attack_response_interactions(defense_results, attack_results)
            
            # Store simulation results
            simulation_result = {
                'timestamp': time.time(),
                'flows_processed': len(flows),
                'attacks_launched': len(attacks),
                'defense_results': defense_results,
                'attack_results': attack_results,
                'interactions': interactions,
                'system_balance': self._calculate_system_balance(defense_results, attack_results)
            }
            
            self.batch_simulations.append(simulation_result)
            
            # Update performance metrics
            self.performance_metrics['total_simulations'] += 1
            self.performance_metrics['last_simulation'] = datetime.now()
            
            logging.info(f"Batch simulation completed: {len(flows)} flows, {len(attacks)} attacks")
            
        except Exception as e:
            logging.error(f"Batch simulation failed: {e}")
    
    def _generate_simulated_flows(self) -> List[NetworkFlow]:
        """Generate simulated network flows and cache them for correlation"""
        flows = []
        
        # Generate normal traffic
        for _ in range(self.config['batch_size'] // 2):
            flow = NetworkFlow(
                src_ip=f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
                dst_ip=f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
                src_port=random.randint(1024, 65535),
                dst_port=random.choice([80, 443, 22, 53, 25]),
                protocol=random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS']),
                packet_count=random.randint(1, 1000),
                byte_count=random.randint(64, 65536),
                duration=random.uniform(0.1, 10.0),
                timestamp=time.time(),
                flags=random.choice(['', 'SYN', 'ACK', 'FIN', 'RST'])
            )
            flows.append(flow)
            # Cache flow for later correlation
            self.flow_cache[flow.flow_id] = flow
        
        # Generate some anomalous traffic (these will be targets for attacks)
        for _ in range(self.config['batch_size'] // 4):
            flow = NetworkFlow(
                src_ip=f"203.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
                dst_ip=f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
                src_port=random.randint(1024, 65535),
                dst_port=random.choice([22, 3389, 1433, 3306]),
                protocol='TCP',
                packet_count=random.randint(1000, 10000),
                byte_count=random.randint(65536, 1048576),
                duration=random.uniform(0.01, 1.0),
                timestamp=time.time(),
                flags='SYN'
            )
            flows.append(flow)
            # Cache flow for later correlation
            self.flow_cache[flow.flow_id] = flow
        
        # Keep cache size bounded
        if len(self.flow_cache) > 10000:
            # Keep only most recent 5000
            recent_ids = list(self.flow_cache.keys())[-5000:]
            self.flow_cache = {k: v for k, v in self.flow_cache.items() if k in recent_ids}
        
        return flows
    
    def _generate_simulated_attacks(self) -> List[Dict[str, Any]]:
        """Generate simulated attacks linked to specific flows"""
        attacks = []
        
        attack_types = list(AttackType)
        
        # Get recent flows from cache that could be attack targets
        # Target the anomalous flows (external IPs attacking internal systems)
        potential_targets = [
            flow for flow in self.flow_cache.values()
            if flow.src_ip.startswith('203.0.') and flow.dst_ip.startswith('10.0.')
        ]
        
        # If no targets available, use any cached flows
        if not potential_targets and self.flow_cache:
            potential_targets = list(self.flow_cache.values())[-20:]  # Last 20 flows
        
        for _ in range(self.config['batch_size'] // 10):
            # Select target flow if available
            target_flow = random.choice(potential_targets) if potential_targets else None
            
            if target_flow:
                # Create attack targeting this specific flow
                attack = {
                    'attack_type': random.choice(attack_types),
                    'target_ip': target_flow.dst_ip,
                    'target_port': target_flow.dst_port,
                    'intensity': random.uniform(0.1, 1.0),
                    'stealth_level': random.randint(1, 10),
                    'target_flow_id': target_flow.flow_id  # Link attack to flow
                }
            else:
                # Fallback to random target
                attack = {
                    'attack_type': random.choice(attack_types),
                    'target_ip': f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
                    'target_port': random.choice([80, 443, 22, 53, 25, 3389, 1433, 3306]),
                    'intensity': random.uniform(0.1, 1.0),
                    'stealth_level': random.randint(1, 10),
                    'target_flow_id': None
                }
            
            attacks.append(attack)
        
        return attacks
    
    def _process_flows_through_order(self, flows: List[NetworkFlow]) -> Dict[str, Any]:
        """Process flows through ORDER engine and track detections"""
        results = {
            'total_flows': len(flows),
            'anomalies_detected': 0,
            'false_positives': 0,
            'true_positives': 0,
            'processing_time': 0.0,
            'signatures_generated': 0,
            'detected_flow_ids': []  # Track which flows were flagged
        }
        
        start_time = time.time()
        
        # Clear previous batch's detections
        self.detected_anomalies.clear()
        
        for flow in flows:
            self.order_engine.process_flow(flow)
        
        # Get results from ORDER engine
        order_status = self.order_engine.get_status()
        anomaly_count_before = results['anomalies_detected']
        results['anomalies_detected'] = order_status['performance_metrics']['anomalies_detected']
        results['processing_time'] = time.time() - start_time
        
        # Track which flows were detected (approximate by checking anomaly signatures)
        # In a real system, ORDER would return the specific flow IDs detected
        recent_signatures = self.order_engine.get_attack_signatures(limit=len(flows))
        for sig in recent_signatures:
            if 'Anomaly_' in sig['name']:
                flow_id = sig['name'].replace('Anomaly_', '')
                self.detected_anomalies.add(flow_id)
                results['detected_flow_ids'].append(flow_id)
        
        return results
    
    def _process_attacks_through_chaos(self, attacks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process attacks through CHAOS engine with proper correlation tracking"""
        results = {
            'total_attacks': len(attacks),
            'successful_attacks': 0,
            'failed_attacks': 0,
            'stealth_maintained': 0,
            'detection_avoided': 0,
            'total_damage': 0,
            'processing_time': 0.0,
            'attack_ids': []
        }
        
        start_time = time.time()
        
        for attack in attacks:
            try:
                attack_id = self.chaos_engine.launch_attack(
                    attack['attack_type'],
                    attack['target_ip'],
                    attack['target_port']
                )
                
                results['attack_ids'].append(attack_id)
                
                # Track attack-flow correlation if target flow exists
                target_flow_id = attack.get('target_flow_id')
                if target_flow_id:
                    if attack_id not in self.attack_target_map:
                        self.attack_target_map[attack_id] = []
                    self.attack_target_map[attack_id].append(target_flow_id)
                    self.flow_attack_map[target_flow_id] = attack_id
                
                # Simulate attack result
                attack_success = random.random() < 0.6
                if attack_success:  # 60% success rate
                    results['successful_attacks'] += 1
                    results['total_damage'] += random.randint(10, 100)
                
                stealth_ok = random.random() < 0.7
                if stealth_ok:  # 70% stealth success
                    results['stealth_maintained'] += 1
                    results['detection_avoided'] += 1
                
                # IMPROVED FEEDBACK: Use actual flow correlation instead of dummy flows
                if self.enable_online_feedback and self.order_engine and self.order_engine.is_trained:
                    if target_flow_id and target_flow_id in self.flow_cache:
                        actual_flow = self.flow_cache[target_flow_id]
                        
                        # Determine ground truth and detection status
                        is_attack = attack_success  # True if attack succeeded
                        was_detected = target_flow_id in self.detected_anomalies
                        
                        # Only send feedback for learning opportunities
                        if is_attack and not was_detected:
                            # FALSE NEGATIVE: Attack succeeded but wasn't detected
                            self.order_engine.submit_feedback(actual_flow, is_attack=True)
                            self.performance_metrics['false_negatives'] += 1
                            logging.info(f"Feedback: False negative on flow {target_flow_id} (attack {attack_id})")
                        elif is_attack and was_detected:
                            # TRUE POSITIVE: Attack detected correctly
                            self.performance_metrics['true_positives'] += 1
                            logging.debug(f"True positive: Attack {attack_id} detected")
                        elif not is_attack and was_detected:
                            # FALSE POSITIVE: Normal flow flagged as attack
                            self.order_engine.submit_feedback(actual_flow, is_attack=False)
                            self.performance_metrics['false_positives'] += 1
                            logging.info(f"Feedback: False positive on flow {target_flow_id}")
                        else:
                            # TRUE NEGATIVE: Normal flow passed correctly
                            self.performance_metrics['true_negatives'] += 1
                
            except Exception as e:
                logging.error(f"Attack processing failed: {e}")
                results['failed_attacks'] += 1
        
        results['processing_time'] = time.time() - start_time
        
        return results
    
    def _track_attack_response_interactions(self, defense_results: Dict[str, Any], attack_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Track interactions between attacks and responses"""
        interactions = []
        
        # Simulate interactions based on results
        for i in range(min(defense_results['anomalies_detected'], attack_results['successful_attacks'])):
            interaction = {
                'interaction_id': f"int_{i}_{int(time.time())}",
                'attack_type': random.choice(list(AttackType)).value,
                'defense_response': random.choice(['blocked', 'detected', 'mitigated', 'ignored']),
                'response_time': random.uniform(0.1, 5.0),
                'effectiveness': random.uniform(0.1, 1.0),
                'timestamp': time.time()
            }
            interactions.append(interaction)
        
        return interactions
    
    def _calculate_system_balance(self, defense_results: Dict[str, Any], attack_results: Dict[str, Any]) -> float:
        """Calculate system balance score"""
        try:
            # Defense effectiveness
            defense_effectiveness = 0.0
            if defense_results['total_flows'] > 0:
                detection_rate = defense_results['anomalies_detected'] / defense_results['total_flows']
                defense_effectiveness = min(1.0, detection_rate * 2)  # Scale up detection rate
            
            # Attack effectiveness
            attack_effectiveness = 0.0
            if attack_results['total_attacks'] > 0:
                success_rate = attack_results['successful_attacks'] / attack_results['total_attacks']
                stealth_rate = attack_results['stealth_maintained'] / attack_results['total_attacks']
                attack_effectiveness = (success_rate + stealth_rate) / 2
            
            # Balance score (higher defense, lower attack = better balance)
            balance_score = defense_effectiveness * (1 - attack_effectiveness)
            
            return balance_score
            
        except Exception as e:
            logging.error(f"Balance calculation failed: {e}")
            return 0.5
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        if self.batch_simulations:
            latest_sim = self.batch_simulations[-1]
            
            if latest_sim['defense_results']['anomalies_detected'] > 0:
                self.performance_metrics['successful_defenses'] += 1
            
            if latest_sim['attack_results']['successful_attacks'] > 0:
                self.performance_metrics['successful_attacks'] += 1
            
            self.performance_metrics['system_balance_score'] = latest_sim['system_balance']
    
    def _should_optimize(self) -> bool:
        """Determine if system optimization is needed"""
        if len(self.batch_simulations) < 5:
            return False
        
        # Check recent balance scores
        recent_scores = [sim['system_balance'] for sim in self.batch_simulations[-5:]]
        if recent_scores:  # Check if recent_scores is not empty
            avg_balance = sum(recent_scores) / len(recent_scores)
        else:
            avg_balance = 0.5  # Default balance score
        
        return avg_balance < self.config['performance_threshold']
    
    def _optimize_system(self):
        """Optimize the system based on performance"""
        try:
            logging.info("Initiating system optimization")
            
            # Get current performance
            order_status = self.order_engine.get_status()
            chaos_status = self.chaos_engine.get_metrics()
            balance_status = self.balance_controller.get_status()
            
            # Adjust system parameters based on performance
            if order_status['performance_metrics']['model_accuracy'] < 0.7:
                logging.info("ORDER engine accuracy low, triggering adaptation")
                # ORDER engine will auto-adapt
            
            if chaos_status['detection_rate'] > 0.5:
                logging.info("CHAOS engine detection rate high, increasing stealth")
                self.chaos_engine.set_stealth_mode(True)
                self.chaos_engine.set_aggression_level(3)
            
            if balance_status['average_reward'] < 0.3:
                logging.info("BALANCE controller reward low, adjusting strategy")
                # BALANCE controller will auto-optimize
            
            logging.info("System optimization completed")
            
        except Exception as e:
            logging.error(f"System optimization failed: {e}")
    
    def _process_real_time_flows(self):
        """Process real-time network flows"""
        # This would integrate with real network monitoring
        # For now, we'll simulate real-time processing
        pass
    
    def _process_real_time_attacks(self):
        """Process real-time attacks"""
        # This would integrate with real attack feeds
        # For now, we'll simulate real-time processing
        pass
    
    def _update_system_state(self):
        """Update system state for real-time mode"""
        # This would update the system state based on real-time data
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            order_status = self.order_engine.get_status()
            chaos_status = self.chaos_engine.get_metrics()
            balance_status = self.balance_controller.get_status()
            
            return {
                'system_running': self.running,
                'simulation_mode': self.simulation_mode,
                'performance_metrics': self.performance_metrics,
                'order_engine': order_status,
                'chaos_engine': chaos_status,
                'balance_controller': balance_status,
                'total_simulations': len(self.batch_simulations),
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def get_simulation_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent simulation results"""
        return self.batch_simulations[-limit:] if self.batch_simulations else []
    
    def get_attack_response_tracking(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get attack-response tracking data"""
        return self.attack_response_tracking[-limit:] if self.attack_response_tracking else []
    
    def save_system_state(self):
        """Save the current system state"""
        try:
            state_data = {
                'performance_metrics': self.performance_metrics,
                'batch_simulations': self.batch_simulations[-100:],  # Keep last 100
                'attack_response_tracking': self.attack_response_tracking[-100:],
                'config': self.config,
                'timestamp': time.time()
            }
            
            state_file = f"{self.config['data_dir']}/system_state.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logging.info("System state saved successfully")
            
        except Exception as e:
            logging.error(f"Failed to save system state: {e}")
    
    def load_system_state(self):
        """Load previously saved system state"""
        try:
            state_file = f"{self.config['data_dir']}/system_state.json"
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.performance_metrics = state_data.get('performance_metrics', self.performance_metrics)
                self.batch_simulations = state_data.get('batch_simulations', [])
                self.attack_response_tracking = state_data.get('attack_response_tracking', [])
                
                logging.info("System state loaded successfully")
                
        except Exception as e:
            logging.error(f"Failed to load system state: {e}")
    
    def shutdown(self):
        """Shutdown the main engine gracefully"""
        try:
            logging.info("Shutting down Self-Morphing AI Cybersecurity Engine...")
            
            self.running = False
            
            # Shutdown components
            if self.order_engine:
                self.order_engine.shutdown()
            
            if self.chaos_engine:
                self.chaos_engine.shutdown()
            
            if self.balance_controller:
                self.balance_controller.shutdown()
            
            # Save final state
            self.save_system_state()
            
            logging.info("Engine shutdown complete")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

def main():
    """Main entry point"""
    try:
        # Initialize and start the engine
        engine = SelfMorphingAICybersecurityEngine()
        
        # Load previous state if available
        engine.load_system_state()
        
        # Start the engine
        engine.start()
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Received interrupt signal")
        
    except Exception as e:
        logging.error(f"Engine failed to start: {e}")
        sys.exit(1)
    finally:
        if 'engine' in locals():
            engine.shutdown()

if __name__ == "__main__":
    main()
