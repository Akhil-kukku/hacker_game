"""
ORDER Engine - Defense System
Self-Morphing AI Cybersecurity Engine - Defense Component
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
from dataclasses import dataclass
import os
import hashlib

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
    
    def __post_init__(self):
        if self.flow_id is None:
            self.flow_id = hashlib.md5(
                f"{self.src_ip}:{self.dst_ip}:{self.src_port}:{self.dst_port}:{self.protocol}".encode()
            ).hexdigest()[:8]

@dataclass
class AttackSignature:
    """Represents an attack signature"""
    name: str
    pattern: str
    confidence: float
    category: str
    timestamp: float
    source: str

class OrderEngine:
    """
    ORDER Defense Engine using Isolation Forest for anomaly detection
    with advanced features for training, preprocessing, logging, and model mutation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        self.attack_signatures = []
        self.flow_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        self.mutation_counter = 0
        self.performance_metrics = {
            'total_flows_processed': 0,
            'anomalies_detected': 0,
            'false_positives': 0,
            'true_positives': 0,
            'model_accuracy': 0.0,
            'last_training_time': None,
            'last_mutation_time': None
        }
        # Online feedback buffers
        self.feedback_buffer: List[Tuple[np.ndarray, int]] = []  # (feature_vector, label)
        self.supervised_mode: bool = False
        self.supervised_threshold: int = 1000
        
        # Initialize model
        self._initialize_model()
        
        # Start processing thread
        self._start_processing()
        
        logging.info("ORDER Engine initialized successfully")

    def enable_supervised_mode(self, enabled: bool = True, threshold: Optional[int] = None):
        """Toggle supervised pretraining mode and optional threshold for training size."""
        self.supervised_mode = enabled
        if threshold is not None:
            self.supervised_threshold = threshold

    def train_from_dataset(self, file_path: str, label_column: Optional[str] = None):
        """Train initial model from a CSV/Parquet dataset. If label_column provided, use supervised prefit of scaler and contamination.

        Expected columns at minimum: packet_count, byte_count, duration, src_port, dst_port, protocol, flags.
        Additional columns are ignored unless label_column is specified for normal/anomaly labels (0=normal,1=anomaly).
        """
        try:
            logging.info(f"Loading dataset from {file_path}")
            if file_path.lower().endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)

            # Build features similar to live extraction
            features: List[List[float]] = []
            labels: List[int] = []

            required = ['packet_count','byte_count','duration','src_port','dst_port','protocol','flags','src_ip','dst_ip']
            for col in required:
                if col not in df.columns:
                    df[col] = 0 if col not in ['protocol','flags','src_ip','dst_ip'] else ''

            for _, row in df.iterrows():
                flow_like = type('F', (), {})()
                flow_like.packet_count = int(row['packet_count'])
                flow_like.byte_count = int(row['byte_count'])
                flow_like.duration = float(row['duration'])
                flow_like.src_port = int(row['src_port'])
                flow_like.dst_port = int(row['dst_port'])
                flow_like.protocol = str(row['protocol'])
                flow_like.flags = str(row['flags'])
                flow_like.src_ip = str(row['src_ip'])
                flow_like.dst_ip = str(row['dst_ip'])
                feat = [
                    flow_like.packet_count,
                    flow_like.byte_count,
                    flow_like.duration,
                    flow_like.src_port,
                    flow_like.dst_port,
                    self._encode_protocol(flow_like.protocol),
                    self._classify_port(flow_like.src_port),
                    self._classify_port(flow_like.dst_port),
                    self._classify_direction(flow_like.src_ip, flow_like.dst_ip),
                    flow_like.packet_count / max(flow_like.duration, 0.001),
                    flow_like.byte_count / max(flow_like.duration, 0.001),
                    self._calculate_entropy([flow_like.src_port, flow_like.dst_port]),
                    self._calculate_flag_complexity(flow_like.flags)
                ]
                features.append(feat)
                if label_column and label_column in df.columns:
                    labels.append(int(row[label_column]))

            X = np.array(features)
            logging.info(f"Prepared features shape: {X.shape}; labels: {len(labels) if labels else 0}")

            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)

            # If labels provided, estimate contamination from label distribution
            if labels:
                anomaly_ratio = max(0.01, min(0.49, float(sum(labels)) / len(labels)))
                self.config['contamination'] = anomaly_ratio
                logging.info(f"Setting contamination to {anomaly_ratio:.3f} based on labels")

            # Train Isolation Forest
            self.model = IsolationForest(
                contamination=self.config['contamination'],
                n_estimators=self.config['n_estimators'],
                max_samples=self.config['max_samples'],
                random_state=self.config['random_state']
            )
            if X_scaled.size == 0 or X_scaled.ndim != 2 or X_scaled.shape[0] < 10:
                raise ValueError(f"Insufficient training samples: {X_scaled.shape if hasattr(X_scaled,'shape') else 'unknown'}")
            self.model.fit(X_scaled)
            self.is_trained = True
            self.training_data = features
            self.performance_metrics['last_training_time'] = datetime.now()
            self._save_model()
            logging.info("Initial training from dataset completed")
        except Exception as e:
            logging.error(f"train_from_dataset failed: {e}")
            raise

    def submit_feedback(self, flow: NetworkFlow, is_attack: bool):
        """Online feedback: push labeled example, periodically refit incremental model surrogate by partial retrain."""
        try:
            feat = self._extract_features([flow])[0]
            label = 1 if is_attack else 0
            self.feedback_buffer.append((feat, label))
            # Trigger lightweight adaptation when buffer is large
            if len(self.feedback_buffer) >= self.supervised_threshold:
                self._apply_feedback_update()
        except Exception as e:
            logging.error(f"submit_feedback failed: {e}")

    def _apply_feedback_update(self):
        """Apply buffered feedback by mixing recent feedback with past training and refitting model."""
        try:
            if not self.feedback_buffer:
                return
            logging.info(f"Applying feedback update with {len(self.feedback_buffer)} samples")
            feedback_X = np.array([x for x, _ in self.feedback_buffer])
            # Merge with a slice of prior training data to stabilize
            recent_hist = np.array(self.training_data[-self.supervised_threshold:]) if self.training_data else np.empty((0, feedback_X.shape[1]))
            X = np.vstack([recent_hist, feedback_X]) if recent_hist.size else feedback_X
            X_scaled = self.scaler.fit_transform(X)
            # Heuristic: set contamination to upper bound of recent anomaly ratio proxy
            est_ratio = 0.1
            if self.feedback_buffer:
                est_ratio = max(0.01, min(0.49, sum(l for _, l in self.feedback_buffer) / len(self.feedback_buffer)))
            self.config['contamination'] = est_ratio
            self.model = IsolationForest(
                contamination=self.config['contamination'],
                n_estimators=self.config['n_estimators'],
                max_samples=self.config['max_samples'],
                random_state=self.config['random_state']
            )
            self.model.fit(X_scaled)
            self.is_trained = True
            # Maintain combined training history bounded
            self.training_data = (self.training_data + [x.tolist() for x in feedback_X])[-max(self.config['training_threshold'], self.supervised_threshold):]
            self.performance_metrics['last_training_time'] = datetime.now()
            self.feedback_buffer.clear()
            self._save_model()
            logging.info("Feedback update applied and model saved")
        except Exception as e:
            logging.error(f"_apply_feedback_update failed: {e}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for ORDER Engine"""
        return {
            'contamination': 0.1,
            'n_estimators': 100,
            'max_samples': 'auto',
            'random_state': 42,
            'batch_size': 1000,
            'training_threshold': 10000,
            'mutation_threshold': 0.8,
            'max_signatures': 1000,
            'confidence_threshold': 0.7,
            'model_save_path': 'models/order_model.pkl',
            'scaler_save_path': 'models/order_scaler.pkl',
            'signatures_save_path': 'data/attack_signatures.json'
        }
    
    def _initialize_model(self):
        """Initialize the Isolation Forest model"""
        try:
            self.model = IsolationForest(
                contamination=self.config['contamination'],
                n_estimators=self.config['n_estimators'],
                max_samples=self.config['max_samples'],
                random_state=self.config['random_state']
            )
            logging.info("Isolation Forest model initialized")
        except Exception as e:
            logging.error(f"Failed to initialize model: {e}")
            raise
    
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
    
    def shutdown(self):
        """Shutdown the ORDER Engine"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        # Save final state
        self._save_model()
        logging.info("ORDER Engine shutdown complete")
