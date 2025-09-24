"""
ORDER Engine - Real-World Defense System
Self-Morphing AI Cybersecurity Engine - Defense Component
Advanced threat detection, network monitoring, and automated response system
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import DBSCAN
import joblib
import json
import logging
import time
import socket
import struct
import threading
import queue
import subprocess
import psutil
import netifaces
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import hashlib
import re
import ipaddress
import dns.resolver
import requests
from collections import defaultdict, deque
import yara
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ORDER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('order_engine.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class NetworkFlow:
    """Represents a network flow for analysis"""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_count: int
    byte_count: int
    duration: float
    timestamp: float
    flags: str
    flow_id: str = None
    payload_hash: str = None
    ttl: int = None
    window_size: int = None
    mss: int = None
    
    def __post_init__(self):
        if self.flow_id is None:
            self.flow_id = hashlib.md5(
                f"{self.src_ip}:{self.dst_ip}:{self.src_port}:{self.dst_port}:{self.protocol}".encode()
            ).hexdigest()[:8]

@dataclass
class ThreatIndicator:
    """Represents a threat indicator (IOC)"""
    indicator_type: str  # IP, Domain, URL, Hash, Email, etc.
    value: str
    threat_type: str
    confidence: float
    source: str
    first_seen: float
    last_seen: float
    tags: List[str] = field(default_factory=list)
    description: str = ""

@dataclass
class AttackSignature:
    """Represents an attack signature"""
    name: str
    pattern: str
    confidence: float
    category: str
    timestamp: float
    source: str
    severity: str = "medium"  # low, medium, high, critical
    mitre_attack_id: str = ""
    cve_references: List[str] = field(default_factory=list)

@dataclass
class SecurityIncident:
    """Represents a security incident"""
    incident_id: str
    title: str
    description: str
    severity: str
    status: str  # open, investigating, contained, resolved
    created_at: float
    updated_at: float
    affected_assets: List[str] = field(default_factory=list)
    indicators: List[ThreatIndicator] = field(default_factory=list)
    response_actions: List[str] = field(default_factory=list)
    assigned_to: str = ""

@dataclass
class NetworkPacket:
    """Represents a captured network packet"""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    payload: bytes
    packet_size: int
    flags: int
    ttl: int
    window_size: int
    packet_id: str = None
    
    def __post_init__(self):
        if self.packet_id is None:
            self.packet_id = hashlib.md5(
                f"{self.timestamp}_{self.src_ip}_{self.dst_ip}_{self.payload}".encode()
            ).hexdigest()[:12]

@dataclass
class SystemProcess:
    """Represents a system process for monitoring"""
    pid: int
    name: str
    cmdline: str
    cpu_percent: float
    memory_percent: float
    create_time: float
    user: str
    parent_pid: int
    status: str
    connections: List[Dict[str, Any]] = field(default_factory=list)

class OrderEngine:
    """
    ORDER Defense Engine - Real-World Cybersecurity Defense System
    Advanced threat detection, network monitoring, and automated response
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # ML Models
        self.anomaly_model = None
        self.classification_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Data storage
        self.training_data = []
        self.attack_signatures = []
        self.threat_indicators = []
        self.security_incidents = []
        self.network_flows = deque(maxlen=10000)
        self.captured_packets = deque(maxlen=5000)
        self.system_processes = {}
        
        # Queues and threads
        self.flow_queue = queue.Queue()
        self.packet_queue = queue.Queue()
        self.threat_intel_queue = queue.Queue()
        self.processing_thread = None
        self.packet_capture_thread = None
        self.threat_intel_thread = None
        self.running = False
        
        # Real-time monitoring
        self.network_interfaces = []
        self.monitored_ports = set()
        self.blocked_ips = set()
        self.whitelist_ips = set()
        self.active_connections = {}
        
        # Threat intelligence
        self.ioc_database = {}
        self.threat_feeds = []
        self.yara_rules = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_flows_processed': 0,
            'packets_captured': 0,
            'threats_detected': 0,
            'incidents_created': 0,
            'false_positives': 0,
            'true_positives': 0,
            'model_accuracy': 0.0,
            'last_training_time': None,
            'last_mutation_time': None,
            'active_threats': 0,
            'blocked_connections': 0
        }
        
        # Initialize components
        self._initialize_models()
        self._initialize_network_monitoring()
        self._initialize_threat_intelligence()
        self._start_processing()
        
        logging.info("ORDER Defense Engine initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for ORDER Engine"""
        return {
            # ML Model settings
            'contamination': 0.1,
            'n_estimators': 100,
            'max_samples': 'auto',
            'random_state': 42,
            'batch_size': 1000,
            'training_threshold': 10000,
            'mutation_threshold': 0.8,
            'confidence_threshold': 0.7,
            
            # Network monitoring
            'monitor_interfaces': True,
            'capture_packets': True,
            'monitor_ports': [22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5900],
            'packet_capture_limit': 1000,
            'flow_timeout': 300,  # seconds
            
            # Threat detection
            'enable_yara': True,
            'enable_dns_monitoring': True,
            'enable_process_monitoring': True,
            'threat_intel_feeds': [
                'https://feeds.feedburner.com/ThreatIntelligence',
                'https://www.malware-traffic-analysis.net/',
                'https://www.abuse.ch/feeds/'
            ],
            
            # Response actions
            'auto_block_threats': True,
            'auto_quarantine': True,
            'create_incidents': True,
            'notify_admins': True,
            
            # File paths
            'model_save_path': 'models/order_model.pkl',
            'scaler_save_path': 'models/order_scaler.pkl',
            'signatures_save_path': 'data/attack_signatures.json',
            'ioc_database_path': 'data/ioc_database.json',
            'incidents_path': 'data/security_incidents.json',
            'yara_rules_path': 'rules/yara_rules.yar',
            
            # Performance
            'max_signatures': 1000,
            'max_incidents': 500,
            'cleanup_interval': 3600,  # 1 hour
            'log_level': 'INFO'
        }
    
    def _initialize_models(self):
        """Initialize ML models for threat detection"""
        try:
            # Anomaly detection model
            self.anomaly_model = IsolationForest(
                contamination=self.config['contamination'],
                n_estimators=self.config['n_estimators'],
                max_samples=self.config['max_samples'],
                random_state=self.config['random_state']
            )
            
            # Classification model for known threats
            self.classification_model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config['random_state']
            )
            
            # Initialize training data storage
            self.training_flows = []
            self.training_labels = []
            self.attack_patterns = []
            
            # Load pre-trained models if available
            self._load_pre_trained_models()
            
            logging.info("ML models initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize models: {e}")
            raise
    
    def _initialize_network_monitoring(self):
        """Initialize network monitoring capabilities"""
        try:
            if self.config['monitor_interfaces']:
                # Get available network interfaces
                self.network_interfaces = netifaces.interfaces()
                logging.info(f"Monitoring interfaces: {self.network_interfaces}")
            
            # Set up monitored ports
            self.monitored_ports = set(self.config['monitor_ports'])
            
            # Initialize connection tracking
            self.active_connections = {}
            
            logging.info("Network monitoring initialized")
        except Exception as e:
            logging.error(f"Failed to initialize network monitoring: {e}")
    
    def _initialize_threat_intelligence(self):
        """Initialize threat intelligence capabilities"""
        try:
            # Load existing IOC database
            self._load_ioc_database()
            
            # Initialize YARA rules if enabled
            if self.config['enable_yara']:
                self._load_yara_rules()
            
            # Start threat intelligence feeds
            self._start_threat_intel_feeds()
            
            logging.info("Threat intelligence initialized")
        except Exception as e:
            logging.error(f"Failed to initialize threat intelligence: {e}")
    
    def _load_ioc_database(self):
        """Load IOC database from file"""
        try:
            if os.path.exists(self.config['ioc_database_path']):
                with open(self.config['ioc_database_path'], 'r') as f:
                    data = json.load(f)
                    self.ioc_database = data
                logging.info(f"Loaded {len(self.ioc_database)} IOCs from database")
        except Exception as e:
            logging.error(f"Failed to load IOC database: {e}")
    
    def _load_yara_rules(self):
        """Load YARA rules for malware detection"""
        try:
            if os.path.exists(self.config['yara_rules_path']):
                self.yara_rules = yara.compile(self.config['yara_rules_path'])
                logging.info("YARA rules loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load YARA rules: {e}")
    
    def _start_threat_intel_feeds(self):
        """Start threat intelligence feed monitoring"""
        if self.config['threat_intel_feeds']:
            self.threat_intel_thread = threading.Thread(
                target=self._monitor_threat_feeds, 
                daemon=True
            )
            self.threat_intel_thread.start()
            logging.info("Threat intelligence feeds started")
    
    def _start_processing(self):
        """Start the background processing thread"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_flows, daemon=True)
        self.processing_thread.start()
        logging.info("Background processing thread started")
    
    def _process_flows(self):
        """Background thread for processing network flows"""
        while self.running:
            try:
                # Process flows in batches
                flows = []
                while len(flows) < self.config['batch_size']:
                    try:
                        flow = self.flow_queue.get(timeout=1)
                        flows.append(flow)
                    except queue.Empty:
                        break
                
                if flows:
                    self._process_batch(flows)
                    
            except Exception as e:
                logging.error(f"Error in flow processing thread: {e}")
                time.sleep(1)
    
    def _process_batch(self, flows: List[NetworkFlow]):
        """Process a batch of network flows"""
        try:
            # Convert flows to feature vectors
            features = self._extract_features(flows)
            
            if not self.is_trained:
                # Store for training
                self.training_data.extend(features)
                if len(self.training_data) >= self.config['training_threshold']:
                    self._train_model()
            else:
                # Detect anomalies
                predictions = self._detect_anomalies(features)
                self._analyze_predictions(flows, predictions)
                
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
    
    def _extract_features(self, flows: List[NetworkFlow]) -> np.ndarray:
        """Extract features from network flows"""
        features = []
        
        for flow in flows:
            # Basic flow features
            flow_features = [
                flow.packet_count,
                flow.byte_count,
                flow.duration,
                flow.src_port,
                flow.dst_port,
                # Protocol encoding (TCP=1, UDP=2, ICMP=3, etc.)
                self._encode_protocol(flow.protocol),
                # Port type (well-known=1, registered=2, dynamic=3)
                self._classify_port(flow.src_port),
                self._classify_port(flow.dst_port),
                # Flow direction (inbound=1, outbound=2)
                self._classify_direction(flow.src_ip, flow.dst_ip),
                # Packet rate
                flow.packet_count / max(flow.duration, 0.001),
                # Byte rate
                flow.byte_count / max(flow.duration, 0.001),
                # Port entropy
                self._calculate_entropy([flow.src_port, flow.dst_port]),
                # Flag complexity
                self._calculate_flag_complexity(flow.flags)
            ]
            features.append(flow_features)
        
        return np.array(features)
    
    def _encode_protocol(self, protocol: str) -> int:
        """Encode protocol string to integer"""
        protocol_map = {
            'TCP': 1, 'UDP': 2, 'ICMP': 3, 'HTTP': 4, 'HTTPS': 5,
            'FTP': 6, 'SSH': 7, 'DNS': 8, 'DHCP': 9, 'SMTP': 10
        }
        return protocol_map.get(protocol.upper(), 0)
    
    def _classify_port(self, port: int) -> int:
        """Classify port type"""
        if port <= 1024:
            return 1  # Well-known
        elif port <= 49151:
            return 2  # Registered
        else:
            return 3  # Dynamic
    
    def _classify_direction(self, src_ip: str, dst_ip: str) -> int:
        """Classify flow direction"""
        # Simple heuristic: assume internal network starts with 10., 192.168., 172.16-31.
        internal_prefixes = ['10.', '192.168.', '172.16.', '172.17.', '172.18.', '172.19.',
                           '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.',
                           '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.']
        
        src_internal = any(src_ip.startswith(prefix) for prefix in internal_prefixes)
        dst_internal = any(dst_ip.startswith(prefix) for prefix in internal_prefixes)
        
        if src_internal and not dst_internal:
            return 1  # Outbound
        elif not src_internal and dst_internal:
            return 2  # Inbound
        else:
            return 3  # Internal
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a list of values"""
        if not values:
            return 0.0
        
        unique_values, counts = np.unique(values, return_counts=True)
        if len(values) > 0:  # Additional safety check
            probabilities = counts / len(values)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy
        else:
            return 0.0
    
    def _calculate_flag_complexity(self, flags: str) -> int:
        """Calculate complexity of TCP flags"""
        if not flags:
            return 0
        
        flag_count = len([f for f in flags if f.isupper()])
        return min(flag_count, 6)  # Max 6 TCP flags
    
    def _train_model(self):
        """Train the Isolation Forest model"""
        try:
            logging.info(f"Training model with {len(self.training_data)} samples")
            
            # Convert to numpy array
            X = np.array(self.training_data)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled)
            self.is_trained = True
            
            # Update metrics
            self.performance_metrics['last_training_time'] = datetime.now()
            self.performance_metrics['total_flows_processed'] += len(self.training_data)
            
            # Save model
            self._save_model()
            
            logging.info("Model training completed successfully")
            
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise
    
    def _detect_anomalies(self, features: np.ndarray) -> np.ndarray:
        """Detect anomalies in feature vectors"""
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict anomalies (-1 for anomaly, 1 for normal)
            predictions = self.model.predict(features_scaled)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}")
            raise
    
    def _analyze_predictions(self, flows: List[NetworkFlow], predictions: np.ndarray):
        """Analyze model predictions and update metrics"""
        try:
            for flow, prediction in zip(flows, predictions):
                self.performance_metrics['total_flows_processed'] += 1
                
                if prediction == -1:  # Anomaly detected
                    self.performance_metrics['anomalies_detected'] += 1
                    
                    # Create attack signature
                    signature = AttackSignature(
                        name=f"Anomaly_{flow.flow_id}",
                        pattern=self._extract_pattern(flow),
                        confidence=self._calculate_confidence(flow),
                        category="Anomaly",
                        timestamp=time.time(),
                        source="ORDER_Engine"
                    )
                    
                    self.attack_signatures.append(signature)
                    
                    # Log anomaly
                    logging.warning(f"Anomaly detected: {flow.flow_id} - {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port}")
                    
                    # Check if mutation is needed
                    if self._should_mutate():
                        self._mutate_model()
            
            # Update accuracy metrics
            self._update_accuracy_metrics()
            
        except Exception as e:
            logging.error(f"Error analyzing predictions: {e}")
    
    def _extract_pattern(self, flow: NetworkFlow) -> str:
        """Extract pattern from network flow"""
        return f"{flow.protocol}:{flow.src_port}:{flow.dst_port}:{flow.packet_count}:{flow.byte_count}"
    
    def _calculate_confidence(self, flow: NetworkFlow) -> float:
        """Calculate confidence score for anomaly detection"""
        # Simple heuristic based on flow characteristics
        confidence = 0.5
        
        # Adjust based on packet count
        if flow.packet_count > 1000:
            confidence += 0.2
        
        # Adjust based on byte count
        if flow.byte_count > 1000000:
            confidence += 0.2
        
        # Adjust based on duration
        if flow.duration < 1.0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _should_mutate(self) -> bool:
        """Determine if model should mutate"""
        if not self.is_trained:
            return False
        
        # Mutate if accuracy drops below threshold
        if self.performance_metrics['model_accuracy'] < self.config['mutation_threshold']:
            return True
        
        # Mutate periodically
        if self.mutation_counter % 1000 == 0:
            return True
        
        return False
    
    def _mutate_model(self):
        """Mutate the model to adapt to new threats"""
        try:
            logging.info("Initiating model mutation")
            
            # Retrain with recent data
            recent_data = self.training_data[-self.config['training_threshold']//2:]
            if len(recent_data) > 0:
                X = np.array(recent_data)
                X_scaled = self.scaler.transform(X)
                self.model.fit(X_scaled)
            
            # Update mutation counter and timestamp
            self.mutation_counter += 1
            self.performance_metrics['last_mutation_time'] = datetime.now()
            
            # Save mutated model
            self._save_model()
            
            logging.info("Model mutation completed")
            
        except Exception as e:
            logging.error(f"Model mutation failed: {e}")
    
    def _update_accuracy_metrics(self):
        """Update accuracy metrics based on recent performance"""
        total = self.performance_metrics['total_flows_processed']
        anomalies = self.performance_metrics['anomalies_detected']
        
        if total > 0:
            # Simple accuracy calculation (can be enhanced with ground truth)
            self.performance_metrics['model_accuracy'] = 1.0 - (anomalies / total)
    
    def _save_model(self):
        """Save the trained model and scaler"""
        try:
            import os
            os.makedirs('models', exist_ok=True)
            os.makedirs('data', exist_ok=True)
            
            # Save model
            joblib.dump(self.model, self.config['model_save_path'])
            
            # Save scaler
            joblib.dump(self.scaler, self.config['scaler_save_path'])
            
            # Save attack signatures
            signatures_data = [
                {
                    'name': sig.name,
                    'pattern': sig.pattern,
                    'confidence': sig.confidence,
                    'category': sig.category,
                    'timestamp': sig.timestamp,
                    'source': sig.source
                }
                for sig in self.attack_signatures[-self.config['max_signatures']:]
            ]
            
            with open(self.config['signatures_save_path'], 'w') as f:
                json.dump(signatures_data, f, indent=2)
            
            logging.info("Model and data saved successfully")
            
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
    
    def load_model(self):
        """Load previously saved model"""
        try:
            if os.path.exists(self.config['model_save_path']):
                self.model = joblib.load(self.config['model_save_path'])
                self.scaler = joblib.load(self.config['scaler_save_path'])
                self.is_trained = True
                logging.info("Model loaded successfully")
            
            if os.path.exists(self.config['signatures_save_path']):
                with open(self.config['signatures_save_path'], 'r') as f:
                    signatures_data = json.load(f)
                
                self.attack_signatures = [
                    AttackSignature(**sig) for sig in signatures_data
                ]
                logging.info(f"Loaded {len(self.attack_signatures)} attack signatures")
                
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
    
    def process_flow(self, flow: NetworkFlow):
        """Process a single network flow"""
        self.flow_queue.put(flow)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of ORDER Engine"""
        return {
            'is_trained': self.is_trained,
            'model_type': 'IsolationForest',
            'performance_metrics': self.performance_metrics.copy(),
            'attack_signatures_count': len(self.attack_signatures),
            'queue_size': self.flow_queue.qsize(),
            'mutation_counter': self.mutation_counter,
            'config': self.config
        }
    
    def get_attack_signatures(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent attack signatures"""
        recent_signatures = self.attack_signatures[-limit:]
        return [
            {
                'name': sig.name,
                'pattern': sig.pattern,
                'confidence': sig.confidence,
                'category': sig.category,
                'timestamp': sig.timestamp,
                'source': sig.source
            }
            for sig in recent_signatures
        ]
    
    def _monitor_threat_feeds(self):
        """Monitor threat intelligence feeds for new IOCs"""
        while self.running:
            try:
                for feed_url in self.config['threat_intel_feeds']:
                    try:
                        response = requests.get(feed_url, timeout=30)
                        if response.status_code == 200:
                            self._process_threat_feed(response.text, feed_url)
                    except Exception as e:
                        logging.error(f"Failed to fetch threat feed {feed_url}: {e}")
                
                # Check feeds every hour
                time.sleep(3600)
            except Exception as e:
                logging.error(f"Error in threat feed monitoring: {e}")
                time.sleep(60)
    
    def _process_threat_feed(self, feed_data: str, source: str):
        """Process threat intelligence feed data"""
        try:
            # Parse different feed formats (CSV, JSON, etc.)
            lines = feed_data.strip().split('\n')
            new_iocs = 0
            
            for line in lines:
                if line.strip():
                    # Simple CSV parsing - can be enhanced for different formats
                    parts = line.split(',')
                    if len(parts) >= 2:
                        ioc_type = parts[0].strip()
                        ioc_value = parts[1].strip()
                        
                        # Create threat indicator
                        indicator = ThreatIndicator(
                            indicator_type=ioc_type,
                            value=ioc_value,
                            threat_type="malware",
                            confidence=0.8,
                            source=source,
                            first_seen=time.time(),
                            last_seen=time.time(),
                            tags=["threat_feed"]
                        )
                        
                        # Add to database
                        key = f"{ioc_type}:{ioc_value}"
                        if key not in self.ioc_database:
                            self.ioc_database[key] = indicator
                            new_iocs += 1
            
            if new_iocs > 0:
                logging.info(f"Added {new_iocs} new IOCs from {source}")
                self._save_ioc_database()
                
        except Exception as e:
            logging.error(f"Failed to process threat feed: {e}")
    
    def _detect_malware(self, payload: bytes) -> List[Dict[str, Any]]:
        """Detect malware using YARA rules"""
        detections = []
        
        try:
            if self.yara_rules:
                matches = self.yara_rules.match(data=payload)
                for match in matches:
                    detection = {
                        'rule_name': match.rule,
                        'tags': match.tags,
                        'meta': match.meta,
                        'confidence': 0.9,
                        'severity': 'high'
                    }
                    detections.append(detection)
                    logging.warning(f"Malware detected: {match.rule}")
        except Exception as e:
            logging.error(f"Malware detection failed: {e}")
        
        return detections
    
    def _check_ioc_match(self, flow: NetworkFlow) -> List[ThreatIndicator]:
        """Check if network flow matches known IOCs"""
        matches = []
        
        try:
            # Check source IP
            src_key = f"IP:{flow.src_ip}"
            if src_key in self.ioc_database:
                matches.append(self.ioc_database[src_key])
            
            # Check destination IP
            dst_key = f"IP:{flow.dst_ip}"
            if dst_key in self.ioc_database:
                matches.append(self.ioc_database[dst_key])
            
            # Check if payload contains known malware hashes
            if flow.payload_hash:
                hash_key = f"Hash:{flow.payload_hash}"
                if hash_key in self.ioc_database:
                    matches.append(self.ioc_database[hash_key])
        
        except Exception as e:
            logging.error(f"IOC matching failed: {e}")
        
        return matches
    
    def _create_security_incident(self, threat_type: str, description: str, 
                                 indicators: List[ThreatIndicator], 
                                 affected_assets: List[str]) -> SecurityIncident:
        """Create a new security incident"""
        incident_id = hashlib.md5(f"{threat_type}_{time.time()}".encode()).hexdigest()[:12]
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"{threat_type} Detected",
            description=description,
            severity="high" if threat_type in ["malware", "apt", "ransomware"] else "medium",
            status="open",
            created_at=time.time(),
            updated_at=time.time(),
            affected_assets=affected_assets,
            indicators=indicators,
            response_actions=[]
        )
        
        self.security_incidents.append(incident)
        self.performance_metrics['incidents_created'] += 1
        
        logging.warning(f"Security incident created: {incident_id}")
        return incident
    
    def _execute_response_action(self, action: str, target: str, incident_id: str = None):
        """Execute automated response actions"""
        try:
            if action == "block_ip":
                self.blocked_ips.add(target)
                logging.info(f"Blocked IP: {target}")
                
            elif action == "quarantine_process":
                # Kill and quarantine suspicious process
                self._quarantine_process(target)
                
            elif action == "isolate_network":
                # Isolate affected network segment
                self._isolate_network_segment(target)
                
            elif action == "notify_admin":
                self._send_admin_notification(incident_id, target)
                
            elif action == "collect_forensics":
                self._collect_forensic_data(target, incident_id)
        
        except Exception as e:
            logging.error(f"Response action failed: {e}")
    
    def _quarantine_process(self, pid: str):
        """Quarantine a suspicious process"""
        try:
            # Kill the process
            subprocess.run(['kill', '-9', pid], check=True)
            logging.info(f"Quarantined process: {pid}")
        except Exception as e:
            logging.error(f"Failed to quarantine process {pid}: {e}")
    
    def _isolate_network_segment(self, network: str):
        """Isolate a network segment"""
        try:
            # Add iptables rule to block network
            cmd = f"iptables -A INPUT -s {network} -j DROP"
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"Isolated network segment: {network}")
        except Exception as e:
            logging.error(f"Failed to isolate network {network}: {e}")
    
    def _send_admin_notification(self, incident_id: str, details: str):
        """Send notification to administrators"""
        try:
            # This would integrate with email/SMS/chat systems
            message = f"Security Alert: Incident {incident_id} - {details}"
            logging.warning(f"ADMIN NOTIFICATION: {message}")
        except Exception as e:
            logging.error(f"Failed to send notification: {e}")
    
    def _collect_forensic_data(self, target: str, incident_id: str):
        """Collect forensic data for investigation"""
        try:
            # Collect network captures, process dumps, etc.
            forensic_dir = f"forensics/incident_{incident_id}"
            os.makedirs(forensic_dir, exist_ok=True)
            
            # Save network flows
            flows_file = f"{forensic_dir}/network_flows.json"
            with open(flows_file, 'w') as f:
                json.dump([flow.__dict__ for flow in self.network_flows], f, indent=2)
            
            logging.info(f"Forensic data collected for incident {incident_id}")
        except Exception as e:
            logging.error(f"Failed to collect forensic data: {e}")
    
    def _save_ioc_database(self):
        """Save IOC database to file"""
        try:
            os.makedirs(os.path.dirname(self.config['ioc_database_path']), exist_ok=True)
            with open(self.config['ioc_database_path'], 'w') as f:
                json.dump(self.ioc_database, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to save IOC database: {e}")
    
    def get_threat_indicators(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent threat indicators"""
        recent_indicators = list(self.ioc_database.values())[-limit:]
        return [
            {
                'indicator_type': ioc.indicator_type,
                'value': ioc.value,
                'threat_type': ioc.threat_type,
                'confidence': ioc.confidence,
                'source': ioc.source,
                'first_seen': ioc.first_seen,
                'last_seen': ioc.last_seen,
                'tags': ioc.tags
            }
            for ioc in recent_indicators
        ]
    
    def get_security_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent security incidents"""
        recent_incidents = self.security_incidents[-limit:]
        return [
            {
                'incident_id': incident.incident_id,
                'title': incident.title,
                'description': incident.description,
                'severity': incident.severity,
                'status': incident.status,
                'created_at': incident.created_at,
                'updated_at': incident.updated_at,
                'affected_assets': incident.affected_assets,
                'indicators_count': len(incident.indicators),
                'response_actions': incident.response_actions
            }
            for incident in recent_incidents
        ]
    
    def block_ip(self, ip: str, reason: str = "Manual block"):
        """Manually block an IP address"""
        self.blocked_ips.add(ip)
        logging.info(f"Manually blocked IP {ip}: {reason}")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)
        logging.info(f"Unblocked IP {ip}")
    
    def add_threat_indicator(self, indicator: ThreatIndicator):
        """Add a new threat indicator"""
        key = f"{indicator.indicator_type}:{indicator.value}"
        self.ioc_database[key] = indicator
        self._save_ioc_database()
        logging.info(f"Added threat indicator: {key}")
    
    def train_with_known_attacks(self, training_data: List[Dict[str, Any]]):
        """Train models with known attack patterns"""
        try:
            logging.info(f"Starting training with {len(training_data)} known attack samples")
            
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if len(X) == 0:
                logging.warning("No valid training data provided")
                return False
            
            # Train anomaly detection model
            self.anomaly_model.fit(X)
            
            # Train classification model
            self.classification_model.fit(X, y)
            
            # Update training status
            self.is_trained = True
            self.performance_metrics['last_training_time'] = time.time()
            
            # Save trained models
            self._save_model()
            
            logging.info("Training completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return False
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from known attacks"""
        X = []
        y = []
        
        for sample in training_data:
            try:
                # Extract features from training sample
                features = self._extract_training_features(sample)
                if features is not None:
                    X.append(features)
                    y.append(sample.get('attack_type', 'unknown'))
            except Exception as e:
                logging.error(f"Failed to process training sample: {e}")
                continue
        
        return np.array(X), np.array(y)
    
    def _extract_training_features(self, sample: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features from training sample"""
        try:
            features = []
            
            # Network flow features
            if 'flow' in sample:
                flow = sample['flow']
                features.extend([
                    flow.get('packet_count', 0),
                    flow.get('byte_count', 0),
                    flow.get('duration', 0),
                    flow.get('src_port', 0),
                    flow.get('dst_port', 0),
                    flow.get('protocol', 0),  # Convert protocol to number
                    flow.get('flags', 0),     # Convert flags to number
                ])
            
            # Attack-specific features
            if 'attack_features' in sample:
                attack_features = sample['attack_features']
                features.extend([
                    attack_features.get('payload_size', 0),
                    attack_features.get('request_rate', 0),
                    attack_features.get('response_time', 0),
                    attack_features.get('error_rate', 0),
                    attack_features.get('connection_count', 0),
                ])
            
            # Behavioral features
            if 'behavior' in sample:
                behavior = sample['behavior']
                features.extend([
                    behavior.get('stealth_level', 0),
                    behavior.get('aggression', 0),
                    behavior.get('complexity', 0),
                    behavior.get('persistence', 0),
                ])
            
            # Ensure we have enough features
            while len(features) < 20:  # Minimum feature count
                features.append(0.0)
            
            return np.array(features[:20])  # Limit to 20 features
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            return None
    
    def _load_pre_trained_models(self):
        """Load pre-trained models if available"""
        try:
            # Load anomaly model
            if os.path.exists(self.config['model_save_path']):
                with open(self.config['model_save_path'], 'rb') as f:
                    model_data = pickle.load(f)
                    self.anomaly_model = model_data.get('anomaly_model', self.anomaly_model)
                    self.classification_model = model_data.get('classification_model', self.classification_model)
                    self.is_trained = model_data.get('is_trained', False)
                logging.info("Pre-trained models loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load pre-trained models: {e}")
    
    def _save_model(self):
        """Save trained models"""
        try:
            os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)
            
            model_data = {
                'anomaly_model': self.anomaly_model,
                'classification_model': self.classification_model,
                'is_trained': self.is_trained,
                'training_time': self.performance_metrics['last_training_time'],
                'scaler': self.scaler
            }
            
            with open(self.config['model_save_path'], 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info("Models saved successfully")
        except Exception as e:
            logging.error(f"Failed to save models: {e}")
    
    def generate_training_data(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate synthetic training data for known attacks"""
        training_data = []
        
        # Known attack types and their characteristics
        attack_types = {
            'ddos': {
                'packet_count': (1000, 10000),
                'byte_count': (100000, 1000000),
                'duration': (0.1, 1.0),
                'src_port': (1024, 65535),
                'dst_port': (80, 443),
                'protocol': 6,  # TCP
                'flags': 2,     # SYN
                'payload_size': (100, 1000),
                'request_rate': (100, 1000),
                'stealth_level': (1, 3),
                'aggression': (8, 10),
                'complexity': (2, 5)
            },
            'sql_injection': {
                'packet_count': (1, 10),
                'byte_count': (100, 10000),
                'duration': (0.5, 5.0),
                'src_port': (1024, 65535),
                'dst_port': (80, 443),
                'protocol': 6,  # TCP
                'flags': 24,    # PSH+ACK
                'payload_size': (200, 2000),
                'request_rate': (1, 10),
                'stealth_level': (4, 7),
                'aggression': (5, 8),
                'complexity': (6, 9)
            },
            'xss': {
                'packet_count': (1, 5),
                'byte_count': (50, 5000),
                'duration': (0.2, 2.0),
                'src_port': (1024, 65535),
                'dst_port': (80, 443),
                'protocol': 6,  # TCP
                'flags': 24,    # PSH+ACK
                'payload_size': (100, 1000),
                'request_rate': (1, 5),
                'stealth_level': (3, 6),
                'aggression': (4, 7),
                'complexity': (4, 7)
            },
            'brute_force': {
                'packet_count': (10, 100),
                'byte_count': (1000, 10000),
                'duration': (1.0, 10.0),
                'src_port': (1024, 65535),
                'dst_port': (22, 23),  # SSH/Telnet
                'protocol': 6,  # TCP
                'flags': 24,    # PSH+ACK
                'payload_size': (50, 200),
                'request_rate': (10, 100),
                'stealth_level': (2, 5),
                'aggression': (6, 9),
                'complexity': (2, 4)
            },
            'normal': {
                'packet_count': (1, 100),
                'byte_count': (100, 10000),
                'duration': (0.1, 5.0),
                'src_port': (1024, 65535),
                'dst_port': (80, 443),
                'protocol': 6,  # TCP
                'flags': 24,    # PSH+ACK
                'payload_size': (100, 1000),
                'request_rate': (1, 10),
                'stealth_level': (5, 8),
                'aggression': (1, 3),
                'complexity': (1, 3)
            }
        }
        
        for _ in range(num_samples):
            attack_type = random.choice(list(attack_types.keys()))
            params = attack_types[attack_type]
            
            # Generate flow data
            flow = {
                'packet_count': random.randint(*params['packet_count']),
                'byte_count': random.randint(*params['byte_count']),
                'duration': random.uniform(*params['duration']),
                'src_port': random.randint(*params['src_port']),
                'dst_port': random.randint(*params['dst_port']),
                'protocol': params['protocol'],
                'flags': params['flags'],
                'src_ip': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                'dst_ip': f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
                'timestamp': time.time()
            }
            
            # Generate attack features
            attack_features = {
                'payload_size': random.randint(*params['payload_size']),
                'request_rate': random.randint(*params['request_rate']),
                'response_time': random.uniform(0.1, 2.0),
                'error_rate': random.uniform(0.0, 0.5),
                'connection_count': random.randint(1, 100)
            }
            
            # Generate behavioral features
            behavior = {
                'stealth_level': random.randint(*params['stealth_level']),
                'aggression': random.randint(*params['aggression']),
                'complexity': random.randint(*params['complexity']),
                'persistence': random.randint(1, 10)
            }
            
            training_sample = {
                'attack_type': attack_type,
                'flow': flow,
                'attack_features': attack_features,
                'behavior': behavior
            }
            
            training_data.append(training_sample)
        
        return training_data
    
    def evaluate_model_performance(self) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        try:
            if not self.is_trained:
                return {"error": "Model not trained"}
            
            # Generate test data
            test_data = self.generate_training_data(100)
            X_test, y_test = self._prepare_training_data(test_data)
            
            if len(X_test) == 0:
                return {"error": "No test data available"}
            
            # Evaluate anomaly detection
            anomaly_scores = self.anomaly_model.decision_function(X_test)
            anomaly_predictions = self.anomaly_model.predict(X_test)
            
            # Evaluate classification
            classification_predictions = self.classification_model.predict(X_test)
            classification_proba = self.classification_model.predict_proba(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, classification_predictions)
            precision = precision_score(y_test, classification_predictions, average='weighted')
            recall = recall_score(y_test, classification_predictions, average='weighted')
            f1 = f1_score(y_test, classification_predictions, average='weighted')
            
            # Anomaly detection metrics
            normal_samples = [i for i, label in enumerate(y_test) if label == 'normal']
            attack_samples = [i for i, label in enumerate(y_test) if label != 'normal']
            
            normal_anomaly_score = np.mean([anomaly_scores[i] for i in normal_samples]) if normal_samples else 0
            attack_anomaly_score = np.mean([anomaly_scores[i] for i in attack_samples]) if attack_samples else 0
            
            performance = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'normal_anomaly_score': normal_anomaly_score,
                'attack_anomaly_score': attack_anomaly_score,
                'anomaly_detection_rate': np.mean(anomaly_predictions == -1),
                'test_samples': len(X_test)
            }
            
            # Update performance metrics
            self.performance_metrics['model_accuracy'] = accuracy
            
            return performance
            
        except Exception as e:
            logging.error(f"Model evaluation failed: {e}")
            return {"error": str(e)}
    
    def shutdown(self):
        """Shutdown the ORDER Engine"""
        self.running = False
        
        # Stop all threads
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        if self.packet_capture_thread:
            self.packet_capture_thread.join(timeout=5)
        if self.threat_intel_thread:
            self.threat_intel_thread.join(timeout=5)
        
        # Save final state
        self._save_model()
        self._save_ioc_database()
        logging.info("ORDER Defense Engine shutdown complete")
