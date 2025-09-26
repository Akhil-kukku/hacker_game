"""
CHAOS Engine - Offensive System
Self-Morphing AI Cybersecurity Engine - Attack Component
"""

import random
import time
import threading
import queue
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import socket
import struct
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CHAOS - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chaos_engine.log'),
        logging.StreamHandler()
    ]
)

class AttackType(Enum):
    """Types of attacks available"""
    DDOS = "DDoS"
    BRUTE_FORCE = "Brute Force"
    SQL_INJECTION = "SQL Injection"
    XSS = "XSS"
    BUFFER_OVERFLOW = "Buffer Overflow"
    MAN_IN_THE_MIDDLE = "Man in the Middle"
    PHISHING = "Phishing"
    MALWARE = "Malware"
    ZERO_DAY = "Zero Day"
    SOCIAL_ENGINEERING = "Social Engineering"
    DNS_AMPLIFICATION = "DNS Amplification"
    ARP_SPOOFING = "ARP Spoofing"
    PING_FLOOD = "Ping Flood"
    SYN_FLOOD = "SYN Flood"
    UDP_FLOOD = "UDP Flood"
    ICMP_FLOOD = "ICMP Flood"
    HTTP_FLOOD = "HTTP Flood"
    SLOWLORIS = "Slowloris"
    HEARTBLEED = "Heartbleed"
    SHELLSHOCK = "Shellshock"

@dataclass
class AttackPayload:
    """Represents an attack payload"""
    attack_type: AttackType
    target_ip: str
    target_port: int
    payload_data: bytes
    signature: str
    timestamp: float
    success_probability: float
    stealth_level: int  # 1-10, higher = more stealthy
    damage_potential: int  # 1-10, higher = more damaging
    complexity: int  # 1-10, higher = more complex

@dataclass
class AttackResult:
    """Result of an attack attempt"""
    attack_id: str
    attack_type: AttackType
    target: str
    success: bool
    response_time: float
    damage_dealt: int
    stealth_maintained: bool
    detection_avoided: bool
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)

class ChaosEngine:
    """
    CHAOS Offensive Engine for simulating diverse and adaptive attacks
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.attack_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        self.attack_history = []
        self.successful_attacks = []
        self.failed_attacks = []
        self.adaptation_counter = 0
        self.stealth_mode = True
        self.aggression_level = 5  # 1-10
        
        # Attack patterns and signatures
        self.attack_patterns = self._initialize_attack_patterns()
        self.evolution_history = []
        
        # Performance metrics
        self.metrics = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'failed_attacks': 0,
            'detection_rate': 0.0,
            'average_damage': 0.0,
            'stealth_success_rate': 0.0,
            'adaptation_count': 0,
            'last_adaptation': None
        }
        # Internal caps for performance
        self._history_cap = self.config.get('max_attack_history', 1000)
        
        # Start processing
        self._start_processing()
        
        logging.info("CHAOS Engine initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for CHAOS Engine"""
        return {
            'max_concurrent_attacks': 5,
            'attack_interval': 1.0,  # seconds
            'stealth_threshold': 0.7,
            'adaptation_threshold': 0.3,
            'max_attack_history': 1000,
            'payload_size_range': (64, 4096),
            'timeout': 30.0,
            'retry_attempts': 3,
            'evolution_rate': 0.1,
            'mutation_probability': 0.2
        }
    
    def _initialize_attack_patterns(self) -> Dict[AttackType, Dict[str, Any]]:
        """Initialize attack patterns and signatures"""
        patterns = {}
        
        for attack_type in AttackType:
            patterns[attack_type] = {
                'base_signature': self._generate_signature(attack_type),
                'success_rate': random.uniform(0.1, 0.9),
                'stealth_level': random.randint(1, 10),
                'damage_potential': random.randint(1, 10),
                'complexity': random.randint(1, 10),
                'adaptation_count': 0,
                'last_used': None,
                'evolution_history': []
            }
        
        return patterns
    
    def _generate_signature(self, attack_type: AttackType) -> str:
        """Generate a unique signature for an attack type"""
        base = f"{attack_type.value}_{random.randint(1000, 9999)}"
        return hashlib.md5(base.encode()).hexdigest()[:16]
    
    def _start_processing(self):
        """Start the attack processing thread"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_attacks, daemon=True)
        self.processing_thread.start()
        logging.info("Attack processing thread started")
    
    def _process_attacks(self):
        """Background thread for processing attacks"""
        while self.running:
            try:
                # Process attacks from queue
                attack = self.attack_queue.get(timeout=1)
                if attack:
                    result = self._execute_attack(attack)
                    self.results_queue.put(result)
                    self._update_metrics(result)
                    
                    # Check if adaptation is needed
                    if self._should_adapt():
                        self._adapt_attack_patterns()
                        
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in attack processing thread: {e}")
                time.sleep(1)
    
    def _execute_attack(self, attack: AttackPayload) -> AttackResult:
        """Execute a single attack"""
        start_time = time.time()
        attack_id = hashlib.md5(f"{attack.attack_type.value}_{start_time}".encode()).hexdigest()[:8]
        
        try:
            logging.info(f"Executing {attack.attack_type.value} attack on {attack.target_ip}:{attack.target_port}")
            
            # Simulate attack execution based on type
            success, damage, stealth_maintained = self._simulate_attack(attack)
            
            response_time = time.time() - start_time
            
            result = AttackResult(
                attack_id=attack_id,
                attack_type=attack.attack_type,
                target=f"{attack.target_ip}:{attack.target_port}",
                success=success,
                response_time=response_time,
                damage_dealt=damage,
                stealth_maintained=stealth_maintained,
                detection_avoided=stealth_maintained,
                timestamp=start_time,
                details={
                    'payload_size': len(attack.payload_data),
                    'signature': attack.signature,
                    'stealth_level': attack.stealth_level,
                    'damage_potential': attack.damage_potential
                }
            )
            
            # Store result
            self.attack_history.append(result)
            if success:
                self.successful_attacks.append(result)
            else:
                self.failed_attacks.append(result)
            
            # Limit history size
            if len(self.attack_history) > self._history_cap:
                self.attack_history = self.attack_history[-self._history_cap:]
            
            return result
            
        except Exception as e:
            logging.error(f"Attack execution failed: {e}")
            return AttackResult(
                attack_id=attack_id,
                attack_type=attack.attack_type,
                target=f"{attack.target_ip}:{attack.target_port}",
                success=False,
                response_time=time.time() - start_time,
                damage_dealt=0,
                stealth_maintained=False,
                detection_avoided=False,
                timestamp=start_time,
                details={'error': str(e)}
            )
    
    def _simulate_attack(self, attack: AttackPayload) -> Tuple[bool, int, bool]:
        """Simulate attack execution with realistic outcomes"""
        
        # Base success probability from attack pattern
        base_success = self.attack_patterns[attack.attack_type]['success_rate']
        
        # Adjust based on stealth level
        stealth_factor = attack.stealth_level / 10.0
        
        # Adjust based on complexity
        complexity_factor = (11 - attack.complexity) / 10.0  # Lower complexity = higher success
        
        # Adjust based on aggression level
        aggression_factor = self.aggression_level / 10.0
        
        # Calculate final success probability
        success_prob = base_success * stealth_factor * complexity_factor * aggression_factor
        success_prob = min(success_prob, 0.95)  # Cap at 95%
        
        # Determine success
        success = random.random() < success_prob
        
        # Calculate damage
        if success:
            base_damage = attack.damage_potential * 10
            damage = random.randint(base_damage // 2, base_damage)
        else:
            damage = random.randint(0, attack.damage_potential)
        
        # Determine stealth maintenance
        stealth_maintained = random.random() < stealth_factor
        
        return success, damage, stealth_maintained
    
    def _should_adapt(self) -> bool:
        """Determine if attack patterns should adapt"""
        if len(self.attack_history) < 10:
            return False
        
        # Calculate recent success rate
        recent_attacks = self.attack_history[-20:]
        if recent_attacks:  # Check if recent_attacks is not empty
            success_rate = sum(1 for a in recent_attacks if a.success) / len(recent_attacks)
        else:
            success_rate = 0.0
        
        # Adapt if success rate is below threshold
        if success_rate < self.config['adaptation_threshold']:
            return True
        
        # Adapt periodically
        if self.adaptation_counter % 50 == 0:
            return True
        
        return False
    
    def _adapt_attack_patterns(self):
        """Adapt attack patterns based on recent performance"""
        try:
            logging.info("Initiating attack pattern adaptation")
            
            # Analyze recent performance
            recent_attacks = self.attack_history[-50:]
            
            for attack_type in AttackType:
                type_attacks = [a for a in recent_attacks if a.attack_type == attack_type]
                
                if type_attacks:
                    # Calculate success rate for this attack type
                    success_rate = sum(1 for a in type_attacks if a.success) / len(type_attacks)
                    
                    # Adjust pattern based on performance
                    pattern = self.attack_patterns[attack_type]
                    
                    if success_rate < 0.3:
                        # Poor performance - increase stealth and complexity
                        pattern['stealth_level'] = min(10, pattern['stealth_level'] + 1)
                        pattern['complexity'] = min(10, pattern['complexity'] + 1)
                        pattern['success_rate'] = min(0.9, pattern['success_rate'] + 0.1)
                    elif success_rate > 0.7:
                        # Good performance - optimize for efficiency
                        pattern['damage_potential'] = min(10, pattern['damage_potential'] + 1)
                        pattern['success_rate'] = min(0.9, pattern['success_rate'] + 0.05)
                    
                    # Record evolution
                    pattern['adaptation_count'] += 1
                    pattern['last_used'] = time.time()
                    pattern['evolution_history'].append({
                        'timestamp': time.time(),
                        'success_rate': success_rate,
                        'stealth_level': pattern['stealth_level'],
                        'damage_potential': pattern['damage_potential'],
                        'complexity': pattern['complexity']
                    })
            
            # Update metrics
            self.adaptation_counter += 1
            self.metrics['adaptation_count'] += 1
            self.metrics['last_adaptation'] = datetime.now()
            
            # Record evolution
            self.evolution_history.append({
                'timestamp': time.time(),
                'adaptation_counter': self.adaptation_counter,
                'overall_success_rate': self.metrics['successful_attacks'] / max(self.metrics['total_attacks'], 1)
            })
            
            logging.info("Attack pattern adaptation completed")
            
        except Exception as e:
            logging.error(f"Attack pattern adaptation failed: {e}")
    
    def _update_metrics(self, result: AttackResult):
        """Update performance metrics"""
        self.metrics['total_attacks'] += 1
        
        if result.success:
            self.metrics['successful_attacks'] += 1
        
        if not result.detection_avoided:
            self.metrics['failed_attacks'] += 1
        
        # Update rates
        total = self.metrics['total_attacks']
        if total > 0:
            self.metrics['detection_rate'] = self.metrics['failed_attacks'] / total
            self.metrics['stealth_success_rate'] = sum(1 for a in self.attack_history if a.stealth_maintained) / total
        
        # Update average damage
        if self.attack_history:  # Check if attack_history is not empty
            self.metrics['average_damage'] = sum(a.damage_dealt for a in self.attack_history) / len(self.attack_history)
        else:
            self.metrics['average_damage'] = 0.0
    
    def launch_attack(self, attack_type: AttackType, target_ip: str, target_port: int = 80) -> str:
        """Launch an attack of specified type"""
        try:
            # Generate payload
            payload_data = self._generate_payload(attack_type)
            
            # Get attack pattern
            pattern = self.attack_patterns[attack_type]
            
            # Create attack payload
            attack = AttackPayload(
                attack_type=attack_type,
                target_ip=target_ip,
                target_port=target_port,
                payload_data=payload_data,
                signature=pattern['base_signature'],
                timestamp=time.time(),
                success_probability=pattern['success_rate'],
                stealth_level=pattern['stealth_level'],
                damage_potential=pattern['damage_potential'],
                complexity=pattern['complexity']
            )
            
            # Queue attack
            self.attack_queue.put(attack)
            
            attack_id = hashlib.md5(f"{attack_type.value}_{attack.timestamp}".encode()).hexdigest()[:8]
            logging.info(f"Queued {attack_type.value} attack (ID: {attack_id})")
            
            return attack_id
            
        except Exception as e:
            logging.error(f"Failed to launch attack: {e}")
            raise
    
    def _generate_payload(self, attack_type: AttackType) -> bytes:
        """Generate payload data for specific attack type"""
        payload_size = random.randint(*self.config['payload_size_range'])
        
        if attack_type == AttackType.DDOS:
            return self._generate_ddos_payload(payload_size)
        elif attack_type == AttackType.SQL_INJECTION:
            return self._generate_sql_injection_payload(payload_size)
        elif attack_type == AttackType.XSS:
            return self._generate_xss_payload(payload_size)
        elif attack_type == AttackType.BRUTE_FORCE:
            return self._generate_brute_force_payload(payload_size)
        elif attack_type == AttackType.BUFFER_OVERFLOW:
            return self._generate_buffer_overflow_payload(payload_size)
        else:
            # Generic payload
            return self._generate_generic_payload(payload_size)
    
    def _generate_ddos_payload(self, size: int) -> bytes:
        """Generate DDoS attack payload"""
        # Simulate various DDoS techniques
        techniques = [
            b"GET / HTTP/1.1\r\nHost: target\r\n\r\n" * (size // 30),
            b"POST /login HTTP/1.1\r\nContent-Length: " + str(size).encode() + b"\r\n\r\n" + b"A" * size,
            b"HEAD / HTTP/1.1\r\nUser-Agent: " + b"X" * size + b"\r\n\r\n"
        ]
        return random.choice(techniques)[:size]
    
    def _generate_sql_injection_payload(self, size: int) -> bytes:
        """Generate SQL injection payload"""
        payloads = [
            b"' OR '1'='1",
            b"'; DROP TABLE users; --",
            b"' UNION SELECT * FROM passwords --",
            b"admin'--",
            b"' OR 1=1#",
            b"' AND (SELECT COUNT(*) FROM users) > 0 --"
        ]
        base_payload = random.choice(payloads)
        return (base_payload * (size // len(base_payload) + 1))[:size]
    
    def _generate_xss_payload(self, size: int) -> bytes:
        """Generate XSS attack payload"""
        payloads = [
            b"<script>alert('XSS')</script>",
            b"<img src=x onerror=alert('XSS')>",
            b"javascript:alert('XSS')",
            b"<svg onload=alert('XSS')>",
            b"<iframe src=javascript:alert('XSS')>"
        ]
        base_payload = random.choice(payloads)
        return (base_payload * (size // len(base_payload) + 1))[:size]
    
    def _generate_brute_force_payload(self, size: int) -> bytes:
        """Generate brute force attack payload"""
        # Simulate password attempts
        passwords = [
            b"admin", b"password", b"123456", b"qwerty", b"letmein",
            b"welcome", b"monkey", b"dragon", b"master", b"football"
        ]
        attempts = []
        for _ in range(size // 20):
            attempts.append(random.choice(passwords))
        return b"\n".join(attempts)[:size]
    
    def _generate_buffer_overflow_payload(self, size: int) -> bytes:
        """Generate buffer overflow payload"""
        # NOP sled + shellcode pattern
        nop_sled = b"\x90" * (size // 2)
        shellcode = b"A" * (size // 4) + b"BBBB" + b"C" * (size // 4)
        return nop_sled + shellcode
    
    def _generate_generic_payload(self, size: int) -> bytes:
        """Generate generic payload"""
        return b"A" * size
    
    def get_attack_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent attack results"""
        recent_results = self.attack_history[-limit:]
        return [
            {
                'attack_id': result.attack_id,
                'attack_type': result.attack_type.value,
                'target': result.target,
                'success': result.success,
                'response_time': result.response_time,
                'damage_dealt': result.damage_dealt,
                'stealth_maintained': result.stealth_maintained,
                'detection_avoided': result.detection_avoided,
                'timestamp': result.timestamp,
                'details': result.details
            }
            for result in recent_results
        ]
    
    def get_attack_patterns(self) -> Dict[str, Any]:
        """Get current attack patterns"""
        return {
            attack_type.value: {
                'success_rate': pattern['success_rate'],
                'stealth_level': pattern['stealth_level'],
                'damage_potential': pattern['damage_potential'],
                'complexity': pattern['complexity'],
                'adaptation_count': pattern['adaptation_count'],
                'last_used': pattern['last_used'],
                'evolution_history_count': len(pattern['evolution_history'])
            }
            for attack_type, pattern in self.attack_patterns.items()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()
    
    def set_aggression_level(self, level: int):
        """Set aggression level (1-10)"""
        self.aggression_level = max(1, min(10, level))
        logging.info(f"Aggression level set to {self.aggression_level}")
    
    def set_stealth_mode(self, enabled: bool):
        """Enable or disable stealth mode"""
        self.stealth_mode = enabled
        logging.info(f"Stealth mode {'enabled' if enabled else 'disabled'}")
    
    def shutdown(self):
        """Shutdown the CHAOS Engine"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logging.info("CHAOS Engine shutdown complete")
