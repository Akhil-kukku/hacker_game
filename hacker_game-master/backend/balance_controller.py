"""
BALANCE Controller - Advanced Self-Morphing AI Orchestration System
Self-Morphing AI Cybersecurity Engine - Evolutionary Control Component
Advanced evolutionary computing, neural networks, and adaptive learning for cybersecurity
"""

import numpy as np
import random
import time
import threading
import queue
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import copy
from collections import deque
import pickle
import os
import math
import statistics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BALANCE - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balance_controller.log'),
        logging.StreamHandler()
    ]
)

class ActionType(Enum):
    """Types of actions the controller can take"""
    ADAPT_DEFENSE = "Adapt Defense"
    EVOLVE_ATTACK = "Evolve Attack"
    BALANCE_STRATEGY = "Balance Strategy"
    MUTATE_BOTH = "Mutate Both"
    OPTIMIZE_PERFORMANCE = "Optimize Performance"
    RESET_SYSTEM = "Reset System"
    ADAPTIVE_LEARNING = "Adaptive Learning"
    GENETIC_EVOLUTION = "Genetic Evolution"

@dataclass
class State:
    """Represents the current state of the system"""
    defense_accuracy: float
    attack_success_rate: float
    system_balance: float
    total_interactions: int
    defense_mutations: int
    attack_adaptations: int
    overall_performance: float
    timestamp: float

@dataclass
class Action:
    """Represents an action taken by the controller"""
    action_type: ActionType
    parameters: Dict[str, Any]
    timestamp: float
    action_id: str = None
    
    def __post_init__(self):
        if self.action_id is None:
            self.action_id = hashlib.md5(f"{self.action_type.value}_{self.timestamp}".encode()).hexdigest()[:8]

@dataclass
class Reward:
    """Represents a reward signal"""
    value: float
    components: Dict[str, float]
    timestamp: float
    description: str

@dataclass
class Experience:
    """Represents a learning experience"""
    state: State
    action: Action
    reward: Reward
    next_state: State
    timestamp: float

@dataclass
class NeuralNetwork:
    """Represents a neural network architecture"""
    layers: List[int]
    activation_functions: List[str]
    optimizer: str
    learning_rate: float
    dropout_rate: float
    batch_size: int
    epochs: int
    model_id: str = None
    
    def __post_init__(self):
        if self.model_id is None:
            self.model_id = hashlib.md5(f"{self.layers}_{time.time()}".encode()).hexdigest()[:8]

@dataclass
class EvolutionaryStrategy:
    """Represents an evolutionary strategy configuration"""
    strategy_type: str  # genetic, differential_evolution, particle_swarm, etc.
    population_size: int
    mutation_rate: float
    crossover_rate: float
    selection_pressure: float
    elitism_rate: float
    diversity_threshold: float
    convergence_threshold: float
    max_generations: int

@dataclass
class AdaptationEvent:
    """Represents a system adaptation event"""
    event_id: str
    event_type: str  # threat_detected, performance_degraded, new_attack_pattern, etc.
    severity: str  # low, medium, high, critical
    description: str
    affected_components: List[str]
    adaptation_actions: List[str]
    timestamp: float
    success: bool = False
    performance_impact: float = 0.0

@dataclass
class ThreatPattern:
    """Represents a learned threat pattern"""
    pattern_id: str
    pattern_type: str
    features: List[float]
    frequency: int
    confidence: float
    first_seen: float
    last_seen: float
    associated_attacks: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)

@dataclass
class SystemGenome:
    """Represents the complete system genome for evolution"""
    defense_genes: Dict[str, Any]
    attack_genes: Dict[str, Any]
    balance_genes: Dict[str, Any]
    neural_genes: Dict[str, Any]
    fitness_score: float = 0.0
    generation: int = 0
    mutations: int = 0
    parent_ids: List[str] = field(default_factory=list)

class GeneticIndividual:
    """Represents an individual in the genetic algorithm"""
    
    def __init__(self, genes: Dict[str, Any]):
        self.genes = genes
        self.fitness = 0.0
        self.age = 0
        self.generation = 0
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the individual's genes"""
        for key, value in self.genes.items():
            if random.random() < mutation_rate:
                if isinstance(value, float):
                    self.genes[key] = value + random.uniform(-0.1, 0.1)
                    self.genes[key] = max(0.0, min(1.0, self.genes[key]))
                elif isinstance(value, int):
                    self.genes[key] = value + random.randint(-1, 1)
                    self.genes[key] = max(1, min(10, self.genes[key]))
    
    def crossover(self, other: 'GeneticIndividual') -> Tuple['GeneticIndividual', 'GeneticIndividual']:
        """Perform crossover with another individual"""
        child1_genes = {}
        child2_genes = {}
        
        for key in self.genes:
            if random.random() < 0.5:
                child1_genes[key] = self.genes[key]
                child2_genes[key] = other.genes[key]
            else:
                child1_genes[key] = other.genes[key]
                child2_genes[key] = self.genes[key]
        
        child1 = GeneticIndividual(child1_genes)
        child2 = GeneticIndividual(child2_genes)
        
        return child1, child2

class BalanceController:
    """
    BALANCE Controller - Advanced Self-Morphing AI Orchestration System
    Advanced evolutionary computing, neural networks, and adaptive learning for cybersecurity
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Core AI components
        self.q_table = {}
        self.experience_buffer = deque(maxlen=self.config['experience_buffer_size'])
        self.epsilon = self.config['initial_epsilon']
        self.learning_rate = self.config['learning_rate']
        self.discount_factor = self.config['discount_factor']
        
        # Advanced AI models
        self.neural_networks = {}
        self.ensemble_models = {}
        self.clustering_models = {}
        self.pattern_recognition_models = {}
        
        # Evolutionary computing
        self.population = []
        self.population_size = self.config['population_size']
        self.generation = 0
        self.best_individual = None
        self.evolutionary_strategy = None
        self.system_genomes = []
        
        # Threat intelligence and pattern recognition
        self.threat_patterns = []
        self.adaptation_events = []
        self.anomaly_detectors = {}
        self.prediction_models = {}
        
        # System state tracking
        self.current_state = None
        self.action_history = []
        self.reward_history = []
        self.performance_history = []
        self.adaptation_history = []
        
        # Control parameters
        self.defense_weight = 0.5
        self.attack_weight = 0.5
        self.balance_threshold = 0.6
        self.adaptation_threshold = 0.7
        self.morphing_enabled = True
        
        # Performance metrics
        self.metrics = {
            'total_actions': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'threat_patterns_learned': 0,
            'neural_networks_trained': 0,
            'system_morphs': 0,
            'average_reward': 0.0,
            'best_fitness': 0.0,
            'generation_count': 0,
            'last_optimization': None,
            'last_morph': None,
            'prediction_accuracy': 0.0,
            'adaptation_success_rate': 0.0
        }
        
        # Initialize advanced components
        self._initialize_neural_networks()
        self._initialize_evolutionary_strategy()
        self._initialize_pattern_recognition()
        self._initialize_genetic_population()
        self._initialize_q_table()
        
        # Start control loop
        self.running = False
        self.control_thread = None
        
        logging.info("Advanced BALANCE Controller initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Advanced BALANCE Controller"""
        return {
            # Core RL settings
            'experience_buffer_size': 10000,
            'initial_epsilon': 0.3,
            'learning_rate': 0.001,
            'discount_factor': 0.95,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            
            # Evolutionary computing
            'population_size': 100,
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'elite_size': 10,
            'generation_limit': 500,
            'fitness_threshold': 0.85,
            'diversity_threshold': 0.1,
            'convergence_threshold': 0.95,
            
            # Neural network settings
            'neural_networks': {
                'threat_detection': {'layers': [64, 32, 16], 'activation': 'relu'},
                'pattern_recognition': {'layers': [128, 64, 32], 'activation': 'tanh'},
                'prediction': {'layers': [32, 16, 8], 'activation': 'sigmoid'},
                'adaptation': {'layers': [48, 24, 12], 'activation': 'relu'}
            },
            'neural_learning_rate': 0.001,
            'neural_epochs': 100,
            'neural_batch_size': 32,
            'neural_dropout': 0.2,
            
            # Pattern recognition
            'pattern_learning_rate': 0.01,
            'pattern_memory_size': 1000,
            'pattern_similarity_threshold': 0.8,
            'pattern_confidence_threshold': 0.7,
            
            # System morphing
            'morphing_enabled': True,
            'morphing_threshold': 0.8,
            'morphing_frequency': 3600,  # 1 hour
            'morphing_intensity': 0.3,
            
            # Control and monitoring
            'control_interval': 2.0,  # seconds
            'optimization_threshold': 0.7,
            'adaptation_threshold': 0.6,
            'prediction_horizon': 300,  # 5 minutes
            
            # File paths
            'save_path': 'models/balance_controller.pkl',
            'neural_models_path': 'models/neural_networks/',
            'patterns_path': 'data/threat_patterns.json',
            'genomes_path': 'data/system_genomes.json',
            
            # Performance monitoring
            'performance_window': 100,
            'adaptation_window': 50,
            'prediction_window': 20,
            
            # Advanced features
            'enable_ensemble_learning': True,
            'enable_transfer_learning': True,
            'enable_meta_learning': True,
            'enable_explainable_ai': True,
            'enable_federated_learning': False
        }
    
    def _initialize_neural_networks(self):
        """Initialize neural networks for different AI tasks"""
        try:
            neural_configs = self.config['neural_networks']
            
            for name, config in neural_configs.items():
                # Create neural network architecture
                nn = NeuralNetwork(
                    layers=config['layers'],
                    activation_functions=[config['activation']] * len(config['layers']),
                    optimizer='adam',
                    learning_rate=self.config['neural_learning_rate'],
                    dropout_rate=self.config['neural_dropout'],
                    batch_size=self.config['neural_batch_size'],
                    epochs=self.config['neural_epochs']
                )
                
                # Create and compile TensorFlow model
                model = self._create_neural_model(nn)
                self.neural_networks[name] = {
                    'architecture': nn,
                    'model': model,
                    'trained': False,
                    'accuracy': 0.0,
                    'last_trained': None
                }
            
            logging.info(f"Initialized {len(self.neural_networks)} neural networks")
        
        except Exception as e:
            logging.error(f"Neural network initialization failed: {e}")
    
    def _create_neural_model(self, nn: NeuralNetwork) -> keras.Model:
        """Create a TensorFlow neural network model"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(nn.layers[0], activation=nn.activation_functions[0], 
                              input_shape=(64,)))  # Assuming 64 input features
        
        # Hidden layers
        for i in range(1, len(nn.layers) - 1):
            model.add(layers.Dense(nn.layers[i], activation=nn.activation_functions[i]))
            model.add(layers.Dropout(nn.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(nn.layers[-1], activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=nn.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _initialize_evolutionary_strategy(self):
        """Initialize evolutionary strategy for system morphing"""
        self.evolutionary_strategy = EvolutionaryStrategy(
            strategy_type='genetic',
            population_size=self.config['population_size'],
            mutation_rate=self.config['mutation_rate'],
            crossover_rate=self.config['crossover_rate'],
            selection_pressure=0.8,
            elitism_rate=0.1,
            diversity_threshold=self.config['diversity_threshold'],
            convergence_threshold=self.config['convergence_threshold'],
            max_generations=self.config['generation_limit']
        )
        
        logging.info("Evolutionary strategy initialized")
    
    def _initialize_pattern_recognition(self):
        """Initialize pattern recognition systems"""
        try:
            # Initialize clustering models for threat pattern recognition
            self.clustering_models['threat_clustering'] = DBSCAN(eps=0.5, min_samples=5)
            self.clustering_models['behavior_clustering'] = KMeans(n_clusters=10, random_state=42)
            
            # Initialize ensemble models
            if self.config['enable_ensemble_learning']:
                self.ensemble_models['threat_classifier'] = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )
                self.ensemble_models['behavior_predictor'] = GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                )
            
            # Initialize anomaly detectors
            self.anomaly_detectors['system_anomaly'] = IsolationForest(contamination=0.1)
            self.anomaly_detectors['network_anomaly'] = IsolationForest(contamination=0.05)
            
            logging.info("Pattern recognition systems initialized")
        
        except Exception as e:
            logging.error(f"Pattern recognition initialization failed: {e}")
    
    def _initialize_genetic_population(self):
        """Initialize the genetic algorithm population with advanced genomes"""
        for _ in range(self.population_size):
            # Create comprehensive system genome
            genome = SystemGenome(
                defense_genes={
                    'adaptation_rate': random.uniform(0.1, 0.9),
                    'sensitivity': random.uniform(0.1, 0.9),
                    'response_time': random.uniform(0.1, 1.0),
                    'learning_rate': random.uniform(0.001, 0.1),
                    'mutation_rate': random.uniform(0.05, 0.3)
                },
                attack_genes={
                    'evolution_rate': random.uniform(0.1, 0.9),
                    'stealth_level': random.uniform(0.1, 1.0),
                    'aggression': random.uniform(0.1, 1.0),
                    'adaptation_speed': random.uniform(0.1, 1.0),
                    'complexity': random.uniform(0.1, 1.0)
                },
                balance_genes={
                    'equilibrium_sensitivity': random.uniform(0.1, 0.9),
                    'adaptation_threshold': random.uniform(0.3, 0.9),
                    'morphing_frequency': random.uniform(0.1, 1.0),
                    'learning_capacity': random.uniform(0.1, 1.0),
                    'stability_factor': random.uniform(0.1, 1.0)
                },
                neural_genes={
                    'network_depth': random.randint(2, 8),
                    'learning_rate': random.uniform(0.0001, 0.01),
                    'regularization': random.uniform(0.0, 0.5),
                    'activation_function': random.choice(['relu', 'tanh', 'sigmoid']),
                    'optimizer': random.choice(['adam', 'sgd', 'rmsprop'])
                }
            )
            
            individual = GeneticIndividual(genome.__dict__)
            self.population.append(individual)
            self.system_genomes.append(genome)
        
        logging.info(f"Advanced genetic population initialized with {self.population_size} individuals")
    
    def _initialize_q_table(self):
        """Initialize the Q-learning table"""
        # Create state-action pairs for all possible combinations
        state_ranges = {
            'defense_accuracy': [0.0, 0.3, 0.6, 0.9],
            'attack_success_rate': [0.0, 0.3, 0.6, 0.9],
            'system_balance': [0.0, 0.3, 0.6, 0.9]
        }
        
        for d_acc in state_ranges['defense_accuracy']:
            for a_succ in state_ranges['attack_success_rate']:
                for s_bal in state_ranges['system_balance']:
                    state_key = f"{d_acc:.1f}_{a_succ:.1f}_{s_bal:.1f}"
                    self.q_table[state_key] = {}
                    for action_type in ActionType:
                        self.q_table[state_key][action_type.value] = 0.0
        
        logging.info("Q-learning table initialized")
    
    def start_control_loop(self):
        """Start the main control loop"""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        logging.info("Control loop started")
    
    def _control_loop(self):
        """Main control loop for orchestrating ORDER and CHAOS"""
        while self.running:
            try:
                # Get current system state
                current_state = self._get_current_state()
                
                # Select action using epsilon-greedy policy
                action = self._select_action(current_state)
                
                # Execute action
                reward = self._execute_action(action)
                
                # Get next state
                next_state = self._get_current_state()
                
                # Store experience
                experience = Experience(
                    state=current_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    timestamp=time.time()
                )
                self.experience_buffer.append(experience)
                
                # Update Q-table
                self._update_q_table(experience)
                
                # Update genetic population
                self._update_genetic_population(reward)
                
                # Check if optimization is needed
                if self._should_optimize():
                    self._optimize_system()
                
                # Update metrics
                self._update_metrics(reward)
                
                # Store action and reward
                self.action_history.append(action)
                self.reward_history.append(reward)
                
                # Decay epsilon
                self.epsilon = max(self.epsilon * self.config['epsilon_decay'], self.config['epsilon_min'])
                
                # Sleep for control interval
                time.sleep(self.config['control_interval'])
                
            except Exception as e:
                logging.error(f"Error in control loop: {e}")
                time.sleep(1)
    
    def _get_current_state(self) -> State:
        """Get the current state of the system"""
        # This would typically get data from ORDER and CHAOS engines
        # For now, we'll simulate the state
        
        defense_accuracy = random.uniform(0.6, 0.9)
        attack_success_rate = random.uniform(0.3, 0.7)
        system_balance = (defense_accuracy + (1 - attack_success_rate)) / 2
        
        state = State(
            defense_accuracy=defense_accuracy,
            attack_success_rate=attack_success_rate,
            system_balance=system_balance,
            total_interactions=self.metrics['total_actions'],
            defense_mutations=random.randint(0, 10),
            attack_adaptations=random.randint(0, 10),
            overall_performance=system_balance,
            timestamp=time.time()
        )
        
        self.current_state = state
        return state
    
    def _select_action(self, state: State) -> Action:
        """Select an action using epsilon-greedy policy"""
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Random action
            action_type = random.choice(list(ActionType))
        else:
            # Best action based on Q-values
            q_values = self.q_table.get(state_key, {})
            if q_values:
                action_type = ActionType(max(q_values, key=q_values.get))
            else:
                action_type = random.choice(list(ActionType))
        
        # Generate action parameters based on genetic individual
        best_individual = self._get_best_individual()
        parameters = self._generate_action_parameters(action_type, best_individual)
        
        action = Action(
            action_type=action_type,
            parameters=parameters,
            timestamp=time.time()
        )
        
        return action
    
    def _state_to_key(self, state: State) -> str:
        """Convert state to Q-table key"""
        d_acc = round(state.defense_accuracy, 1)
        a_succ = round(state.attack_success_rate, 1)
        s_bal = round(state.system_balance, 1)
        return f"{d_acc:.1f}_{a_succ:.1f}_{s_bal:.1f}"
    
    def _generate_action_parameters(self, action_type: ActionType, individual: GeneticIndividual) -> Dict[str, Any]:
        """Generate parameters for an action based on genetic individual"""
        if action_type == ActionType.ADAPT_DEFENSE:
            return {
                'adaptation_rate': individual.genes['defense_adaptation_rate'],
                'mutation_threshold': individual.genes['mutation_threshold'],
                'learning_rate': individual.genes['learning_rate']
            }
        elif action_type == ActionType.EVOLVE_ATTACK:
            return {
                'evolution_rate': individual.genes['attack_evolution_rate'],
                'aggression_level': individual.genes['aggression_level'],
                'stealth_preference': individual.genes['stealth_preference']
            }
        elif action_type == ActionType.BALANCE_STRATEGY:
            return {
                'balance_sensitivity': individual.genes['balance_sensitivity'],
                'defense_weight': self.defense_weight,
                'attack_weight': self.attack_weight
            }
        elif action_type == ActionType.MUTATE_BOTH:
            return {
                'defense_mutation_rate': individual.genes['defense_adaptation_rate'],
                'attack_mutation_rate': individual.genes['attack_evolution_rate'],
                'synchronization_factor': random.uniform(0.5, 1.0)
            }
        else:
            return {
                'intensity': random.uniform(0.1, 1.0),
                'duration': random.randint(1, 10),
                'target_component': random.choice(['defense', 'attack', 'both'])
            }
    
    def _execute_action(self, action: Action) -> Reward:
        """Execute an action and return the reward"""
        try:
            logging.info(f"Executing action: {action.action_type.value}")
            
            # Simulate action execution
            success = random.random() < 0.8  # 80% success rate
            
            if success:
                # Calculate reward based on action type and parameters
                reward_value = self._calculate_reward(action)
                components = {
                    'action_success': 1.0,
                    'system_improvement': random.uniform(0.1, 0.5),
                    'balance_maintenance': random.uniform(0.1, 0.3)
                }
            else:
                reward_value = -0.5
                components = {
                    'action_success': 0.0,
                    'system_improvement': -0.2,
                    'balance_maintenance': -0.3
                }
            
            reward = Reward(
                value=reward_value,
                components=components,
                timestamp=time.time(),
                description=f"{action.action_type.value} {'succeeded' if success else 'failed'}"
            )
            
            return reward
            
        except Exception as e:
            logging.error(f"Action execution failed: {e}")
            return Reward(
                value=-1.0,
                components={'error': -1.0},
                timestamp=time.time(),
                description=f"Action execution error: {e}"
            )
    
    def _calculate_reward(self, action: Action) -> float:
        """Calculate reward for an action"""
        base_reward = 0.0
        
        if action.action_type == ActionType.ADAPT_DEFENSE:
            base_reward = 0.3
        elif action.action_type == ActionType.EVOLVE_ATTACK:
            base_reward = 0.2
        elif action.action_type == ActionType.BALANCE_STRATEGY:
            base_reward = 0.4
        elif action.action_type == ActionType.MUTATE_BOTH:
            base_reward = 0.5
        else:
            base_reward = 0.1
        
        # Adjust based on parameters
        if 'adaptation_rate' in action.parameters:
            base_reward *= action.parameters['adaptation_rate']
        if 'evolution_rate' in action.parameters:
            base_reward *= action.parameters['evolution_rate']
        if 'balance_sensitivity' in action.parameters:
            base_reward *= action.parameters['balance_sensitivity']
        
        return base_reward
    
    def _update_q_table(self, experience: Experience):
        """Update Q-table using Q-learning"""
        try:
            current_state_key = self._state_to_key(experience.state)
            next_state_key = self._state_to_key(experience.next_state)
            
            # Get current Q-value
            current_q = self.q_table.get(current_state_key, {}).get(experience.action.action_type.value, 0.0)
            
            # Get max Q-value for next state
            next_q_values = self.q_table.get(next_state_key, {})
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
            
            # Q-learning update
            new_q = current_q + self.learning_rate * (
                experience.reward.value + self.discount_factor * max_next_q - current_q
            )
            
            # Update Q-table
            if current_state_key not in self.q_table:
                self.q_table[current_state_key] = {}
            self.q_table[current_state_key][experience.action.action_type.value] = new_q
            
        except Exception as e:
            logging.error(f"Q-table update failed: {e}")
    
    def _update_genetic_population(self, reward: Reward):
        """Update genetic population based on reward"""
        try:
            # Update fitness of current best individual
            if self.best_individual:
                self.best_individual.fitness += reward.value
                self.best_individual.age += 1
            
            # Periodically evolve population
            if len(self.reward_history) % 10 == 0:
                self._evolve_population()
                
        except Exception as e:
            logging.error(f"Genetic population update failed: {e}")
    
    def _evolve_population(self):
        """Evolve the genetic population"""
        try:
            logging.info("Evolving genetic population")
            
            # Calculate fitness for all individuals
            for individual in self.population:
                individual.fitness = self._calculate_fitness(individual)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Keep elite individuals
            elite_size = self.config['elite_size']
            elite = self.population[:elite_size]
            
            # Create new population
            new_population = elite.copy()
            
            # Generate offspring through crossover
            while len(new_population) < self.population_size:
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                
                if random.random() < self.config['crossover_rate']:
                    child1, child2 = parent1.crossover(parent2)
                    new_population.extend([child1, child2])
                else:
                    # Clone parents
                    child1 = GeneticIndividual(copy.deepcopy(parent1.genes))
                    child2 = GeneticIndividual(copy.deepcopy(parent2.genes))
                    new_population.extend([child1, child2])
            
            # Trim to population size
            new_population = new_population[:self.population_size]
            
            # Mutate non-elite individuals
            for individual in new_population[elite_size:]:
                individual.mutate(self.config['mutation_rate'])
                individual.generation = self.generation + 1
            
            # Update population
            self.population = new_population
            self.generation += 1
            
            # Update best individual
            self.best_individual = self.population[0]
            self.metrics['best_fitness'] = self.best_individual.fitness
            self.metrics['generation_count'] = self.generation
            
            logging.info(f"Population evolved to generation {self.generation}")
            
        except Exception as e:
            logging.error(f"Population evolution failed: {e}")
    
    def _calculate_fitness(self, individual: GeneticIndividual) -> float:
        """Calculate fitness of a genetic individual"""
        # Base fitness from genes
        fitness = 0.0
        
        # Defense adaptation rate (higher is better)
        fitness += individual.genes['defense_adaptation_rate'] * 0.2
        
        # Attack evolution rate (moderate is better)
        attack_rate = individual.genes['attack_evolution_rate']
        fitness += (1 - abs(attack_rate - 0.5)) * 0.2
        
        # Balance sensitivity (higher is better)
        fitness += individual.genes['balance_sensitivity'] * 0.2
        
        # Learning rate (moderate is better)
        learning_rate = individual.genes['learning_rate']
        fitness += (1 - abs(learning_rate - 0.1)) * 0.1
        
        # Stealth preference (moderate is better)
        stealth_pref = individual.genes['stealth_preference']
        fitness += (1 - abs(stealth_pref - 0.5)) * 0.1
        
        # Age penalty
        age_penalty = individual.age * 0.01
        fitness -= age_penalty
        
        return max(0.0, fitness)
    
    def _select_parent(self) -> GeneticIndividual:
        """Select a parent using tournament selection"""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _get_best_individual(self) -> GeneticIndividual:
        """Get the best individual from the population"""
        if not self.population:
            return None
        
        return max(self.population, key=lambda x: x.fitness)
    
    def _should_optimize(self) -> bool:
        """Determine if system optimization is needed"""
        if len(self.reward_history) < 10:
            return False
        
        # Check recent performance
        recent_rewards = [r.value for r in self.reward_history[-10:]]
        if recent_rewards:  # Check if recent_rewards is not empty
            avg_reward = sum(recent_rewards) / len(recent_rewards)
        else:
            avg_reward = 0.0
        
        return avg_reward < self.config['optimization_threshold']
    
    def _optimize_system(self):
        """Optimize the system based on learned patterns"""
        try:
            logging.info("Initiating system optimization")
            
            # Adjust weights based on performance
            if self.current_state:
                if self.current_state.defense_accuracy < 0.7:
                    self.defense_weight = min(0.8, self.defense_weight + 0.1)
                if self.current_state.attack_success_rate > 0.6:
                    self.attack_weight = max(0.2, self.attack_weight - 0.1)
            
            # Update balance threshold
            if self.current_state and self.current_state.system_balance < 0.5:
                self.balance_threshold = max(0.4, self.balance_threshold - 0.1)
            
            self.metrics['last_optimization'] = datetime.now()
            logging.info("System optimization completed")
            
        except Exception as e:
            logging.error(f"System optimization failed: {e}")
    
    def _update_metrics(self, reward: Reward):
        """Update performance metrics"""
        self.metrics['total_actions'] += 1
        
        if reward.value > 0:
            self.metrics['successful_adaptations'] += 1
        else:
            self.metrics['failed_adaptations'] += 1
        
        # Update average reward
        if self.reward_history:  # Check if reward_history is not empty
            total_rewards = sum(r.value for r in self.reward_history)
            self.metrics['average_reward'] = total_rewards / len(self.reward_history)
        else:
            self.metrics['average_reward'] = 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of BALANCE Controller"""
        return {
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.metrics['best_fitness'],
            'average_reward': self.metrics['average_reward'],
            'defense_weight': self.defense_weight,
            'attack_weight': self.attack_weight,
            'balance_threshold': self.balance_threshold,
            'metrics': self.metrics.copy(),
            'current_state': {
                'defense_accuracy': self.current_state.defense_accuracy if self.current_state else 0.0,
                'attack_success_rate': self.current_state.attack_success_rate if self.current_state else 0.0,
                'system_balance': self.current_state.system_balance if self.current_state else 0.0
            } if self.current_state else {}
        }
    
    def get_action_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent action history"""
        recent_actions = self.action_history[-limit:]
        return [
            {
                'action_type': action.action_type.value,
                'parameters': action.parameters,
                'timestamp': action.timestamp,
                'action_id': action.action_id
            }
            for action in recent_actions
        ]
    
    def get_reward_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent reward history"""
        recent_rewards = self.reward_history[-limit:]
        return [
            {
                'value': reward.value,
                'components': reward.components,
                'timestamp': reward.timestamp,
                'description': reward.description
            }
            for reward in recent_rewards
        ]
    
    def save_state(self):
        """Save the current state of the controller"""
        try:
            os.makedirs('models', exist_ok=True)
            
            state_data = {
                'q_table': self.q_table,
                'population': self.population,
                'generation': self.generation,
                'best_individual': self.best_individual,
                'metrics': self.metrics,
                'defense_weight': self.defense_weight,
                'attack_weight': self.attack_weight,
                'balance_threshold': self.balance_threshold,
                'epsilon': self.epsilon,
                'learning_rate': self.learning_rate
            }
            
            with open(self.config['save_path'], 'wb') as f:
                pickle.dump(state_data, f)
            
            logging.info("Controller state saved successfully")
            
        except Exception as e:
            logging.error(f"Failed to save controller state: {e}")
    
    def load_state(self):
        """Load previously saved state"""
        try:
            if os.path.exists(self.config['save_path']):
                with open(self.config['save_path'], 'rb') as f:
                    state_data = pickle.load(f)
                
                self.q_table = state_data['q_table']
                self.population = state_data['population']
                self.generation = state_data['generation']
                self.best_individual = state_data['best_individual']
                self.metrics = state_data['metrics']
                self.defense_weight = state_data['defense_weight']
                self.attack_weight = state_data['attack_weight']
                self.balance_threshold = state_data['balance_threshold']
                self.epsilon = state_data['epsilon']
                self.learning_rate = state_data['learning_rate']
                
                logging.info("Controller state loaded successfully")
                
        except Exception as e:
            logging.error(f"Failed to load controller state: {e}")
    
    def _learn_threat_pattern(self, features: List[float], attack_type: str, confidence: float):
        """Learn new threat patterns from observed attacks"""
        try:
            # Check if similar pattern already exists
            existing_pattern = self._find_similar_pattern(features)
            
            if existing_pattern:
                # Update existing pattern
                existing_pattern.frequency += 1
                existing_pattern.last_seen = time.time()
                existing_pattern.confidence = max(existing_pattern.confidence, confidence)
                if attack_type not in existing_pattern.associated_attacks:
                    existing_pattern.associated_attacks.append(attack_type)
            else:
                # Create new pattern
                pattern = ThreatPattern(
                    pattern_id=hashlib.md5(f"{features}_{time.time()}".encode()).hexdigest()[:8],
                    pattern_type=attack_type,
                    features=features,
                    frequency=1,
                    confidence=confidence,
                    first_seen=time.time(),
                    last_seen=time.time(),
                    associated_attacks=[attack_type]
                )
                self.threat_patterns.append(pattern)
                self.metrics['threat_patterns_learned'] += 1
            
            # Train neural networks on new pattern
            self._train_pattern_recognition_models()
            
            logging.info(f"Learned new threat pattern: {attack_type}")
        
        except Exception as e:
            logging.error(f"Threat pattern learning failed: {e}")
    
    def _find_similar_pattern(self, features: List[float]) -> Optional[ThreatPattern]:
        """Find similar existing threat pattern"""
        for pattern in self.threat_patterns:
            similarity = self._calculate_pattern_similarity(features, pattern.features)
            if similarity > self.config['pattern_similarity_threshold']:
                return pattern
        return None
    
    def _calculate_pattern_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Calculate similarity between two feature vectors"""
        if len(features1) != len(features2):
            return 0.0
        
        # Use cosine similarity
        dot_product = sum(a * b for a, b in zip(features1, features2))
        magnitude1 = math.sqrt(sum(a * a for a in features1))
        magnitude2 = math.sqrt(sum(b * b for b in features2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _train_pattern_recognition_models(self):
        """Train neural networks and ensemble models on threat patterns"""
        try:
            if not self.threat_patterns:
                return
            
            # Prepare training data
            X = np.array([pattern.features for pattern in self.threat_patterns])
            y = [pattern.pattern_type for pattern in self.threat_patterns]
            
            # Train neural networks
            for name, nn_data in self.neural_networks.items():
                if name == 'pattern_recognition':
                    # Convert to categorical
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    y_categorical = keras.utils.to_categorical(y_encoded)
                    
                    # Train model
                    nn_data['model'].fit(X, y_categorical, epochs=10, batch_size=32, verbose=0)
                    nn_data['trained'] = True
                    nn_data['last_trained'] = time.time()
                    self.metrics['neural_networks_trained'] += 1
            
            # Train ensemble models
            if self.config['enable_ensemble_learning']:
                self.ensemble_models['threat_classifier'].fit(X, y)
                self.ensemble_models['behavior_predictor'].fit(X, y)
            
            logging.info("Pattern recognition models trained")
        
        except Exception as e:
            logging.error(f"Pattern recognition training failed: {e}")
    
    def _morph_system(self, adaptation_event: AdaptationEvent):
        """Morph the system based on adaptation event"""
        try:
            if not self.config['morphing_enabled']:
                return
            
            logging.info(f"System morphing triggered: {adaptation_event.event_type}")
            
            # Determine morphing strategy based on event type
            if adaptation_event.event_type == "threat_detected":
                self._morph_defense_systems(adaptation_event)
            elif adaptation_event.event_type == "performance_degraded":
                self._morph_performance_systems(adaptation_event)
            elif adaptation_event.event_type == "new_attack_pattern":
                self._morph_learning_systems(adaptation_event)
            
            # Update system genome
            self._evolve_system_genome(adaptation_event)
            
            # Record morphing event
            self.metrics['system_morphs'] += 1
            self.metrics['last_morph'] = time.time()
            
            logging.info("System morphing completed")
        
        except Exception as e:
            logging.error(f"System morphing failed: {e}")
    
    def _morph_defense_systems(self, event: AdaptationEvent):
        """Morph defense systems based on threat"""
        try:
            # Adjust defense parameters
            self.defense_weight = min(0.9, self.defense_weight + 0.1)
            self.balance_threshold = max(0.4, self.balance_threshold - 0.1)
            
            # Retrain neural networks with new threat data
            for name, nn_data in self.neural_networks.items():
                if name == 'threat_detection':
                    # Add new training data and retrain
                    self._retrain_neural_network(nn_data, event)
            
            logging.info("Defense systems morphed")
        
        except Exception as e:
            logging.error(f"Defense system morphing failed: {e}")
    
    def _morph_performance_systems(self, event: AdaptationEvent):
        """Morph performance systems based on degradation"""
        try:
            # Adjust learning rates
            self.learning_rate = min(0.01, self.learning_rate * 1.1)
            
            # Optimize neural network architectures
            for name, nn_data in self.neural_networks.items():
                if not nn_data['trained'] or nn_data['accuracy'] < 0.7:
                    self._optimize_neural_architecture(nn_data)
            
            logging.info("Performance systems morphed")
        
        except Exception as e:
            logging.error(f"Performance system morphing failed: {e}")
    
    def _morph_learning_systems(self, event: AdaptationEvent):
        """Morph learning systems based on new attack patterns"""
        try:
            # Increase pattern learning rate
            self.config['pattern_learning_rate'] = min(0.1, self.config['pattern_learning_rate'] * 1.2)
            
            # Retrain clustering models
            if self.threat_patterns:
                X = np.array([pattern.features for pattern in self.threat_patterns])
                self.clustering_models['threat_clustering'].fit(X)
            
            logging.info("Learning systems morphed")
        
        except Exception as e:
            logging.error(f"Learning system morphing failed: {e}")
    
    def _evolve_system_genome(self, event: AdaptationEvent):
        """Evolve the system genome based on adaptation event"""
        try:
            # Select best performing genome
            if self.system_genomes:
                best_genome = max(self.system_genomes, key=lambda g: g.fitness_score)
                
                # Create evolved genome
                evolved_genome = self._create_evolved_genome(best_genome, event)
                self.system_genomes.append(evolved_genome)
                
                # Limit genome population
                if len(self.system_genomes) > self.config['population_size']:
                    self.system_genomes = sorted(self.system_genomes, 
                                               key=lambda g: g.fitness_score, reverse=True)[:self.config['population_size']]
            
            logging.info("System genome evolved")
        
        except Exception as e:
            logging.error(f"Genome evolution failed: {e}")
    
    def _create_evolved_genome(self, parent_genome: SystemGenome, event: AdaptationEvent) -> SystemGenome:
        """Create evolved genome from parent"""
        # Create mutated copy
        evolved_genome = copy.deepcopy(parent_genome)
        evolved_genome.generation = parent_genome.generation + 1
        evolved_genome.mutations += 1
        evolved_genome.parent_ids = [parent_genome.defense_genes.get('id', 'unknown')]
        
        # Apply mutations based on event type
        mutation_rate = self.config['mutation_rate']
        
        for gene_type in ['defense_genes', 'attack_genes', 'balance_genes', 'neural_genes']:
            genes = getattr(evolved_genome, gene_type)
            for key, value in genes.items():
                if random.random() < mutation_rate:
                    if isinstance(value, float):
                        genes[key] = max(0.0, min(1.0, value + random.uniform(-0.1, 0.1)))
                    elif isinstance(value, int):
                        genes[key] = max(1, value + random.randint(-1, 1))
        
        return evolved_genome
    
    def _retrain_neural_network(self, nn_data: Dict[str, Any], event: AdaptationEvent):
        """Retrain neural network with new data"""
        try:
            # This would retrain the neural network with new threat data
            # For now, we'll simulate the retraining
            nn_data['trained'] = True
            nn_data['last_trained'] = time.time()
            nn_data['accuracy'] = min(1.0, nn_data['accuracy'] + 0.05)
        
        except Exception as e:
            logging.error(f"Neural network retraining failed: {e}")
    
    def _optimize_neural_architecture(self, nn_data: Dict[str, Any]):
        """Optimize neural network architecture"""
        try:
            # This would optimize the neural network architecture
            # For now, we'll simulate the optimization
            nn_data['accuracy'] = min(1.0, nn_data['accuracy'] + 0.1)
        
        except Exception as e:
            logging.error(f"Neural architecture optimization failed: {e}")
    
    def predict_threat(self, features: List[float]) -> Dict[str, Any]:
        """Predict threat type using trained models"""
        try:
            predictions = {}
            
            # Neural network prediction
            for name, nn_data in self.neural_networks.items():
                if nn_data['trained']:
                    X = np.array([features])
                    pred = nn_data['model'].predict(X)
                    predictions[f'neural_{name}'] = pred.tolist()
            
            # Ensemble prediction
            if self.ensemble_models and self.threat_patterns:
                X = np.array([features])
                for name, model in self.ensemble_models.items():
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                        predictions[f'ensemble_{name}'] = pred.tolist()
            
            # Pattern matching
            similar_pattern = self._find_similar_pattern(features)
            if similar_pattern:
                predictions['pattern_match'] = {
                    'pattern_id': similar_pattern.pattern_id,
                    'confidence': similar_pattern.confidence,
                    'pattern_type': similar_pattern.pattern_type
                }
            
            return predictions
        
        except Exception as e:
            logging.error(f"Threat prediction failed: {e}")
            return {}
    
    def get_adaptation_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent adaptation events"""
        recent_events = self.adaptation_events[-limit:]
        return [
            {
                'event_id': event.event_id,
                'event_type': event.event_type,
                'severity': event.severity,
                'description': event.description,
                'affected_components': event.affected_components,
                'adaptation_actions': event.adaptation_actions,
                'timestamp': event.timestamp,
                'success': event.success,
                'performance_impact': event.performance_impact
            }
            for event in recent_events
        ]
    
    def get_threat_patterns(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get learned threat patterns"""
        recent_patterns = self.threat_patterns[-limit:]
        return [
            {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'frequency': pattern.frequency,
                'confidence': pattern.confidence,
                'first_seen': pattern.first_seen,
                'last_seen': pattern.last_seen,
                'associated_attacks': pattern.associated_attacks,
                'mitigation_strategies': pattern.mitigation_strategies
            }
            for pattern in recent_patterns
        ]
    
    def get_system_genomes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get system genomes"""
        recent_genomes = self.system_genomes[-limit:]
        return [
            {
                'generation': genome.generation,
                'fitness_score': genome.fitness_score,
                'mutations': genome.mutations,
                'parent_ids': genome.parent_ids,
                'defense_genes': genome.defense_genes,
                'attack_genes': genome.attack_genes,
                'balance_genes': genome.balance_genes,
                'neural_genes': genome.neural_genes
            }
            for genome in recent_genomes
        ]
    
    def train_with_known_scenarios(self, training_data: List[Dict[str, Any]]):
        """Train the BALANCE controller with known attack-defense scenarios"""
        try:
            logging.info(f"Starting BALANCE training with {len(training_data)} scenarios")
            
            # Process training scenarios
            for scenario in training_data:
                state = scenario.get('state', {})
                action = scenario.get('action', {})
                reward = scenario.get('reward', 0.0)
                next_state = scenario.get('next_state', {})
                
                # Create experience
                experience = Experience(
                    state=State(**state),
                    action=Action(**action),
                    reward=Reward(value=reward, components={}, timestamp=time.time(), description="Training"),
                    next_state=State(**next_state),
                    timestamp=time.time()
                )
                
                # Add to experience buffer
                self.experience_buffer.append(experience)
                
                # Update Q-table
                self._update_q_table_from_experience(experience)
            
            # Train neural networks
            self._train_neural_networks_from_experiences()
            
            # Evolve genetic population
            self._evolve_population_from_training()
            
            logging.info("BALANCE training completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"BALANCE training failed: {e}")
            return False
    
    def _update_q_table_from_experience(self, experience: Experience):
        """Update Q-table from training experience"""
        try:
            state_key = self._state_to_key(experience.state)
            action_key = f"{experience.action.action_type}_{experience.action.parameters}"
            
            # Q-learning update
            current_q = self.q_table.get((state_key, action_key), 0.0)
            next_state_key = self._state_to_key(experience.next_state)
            
            # Find max Q-value for next state
            max_next_q = 0.0
            for (s_key, a_key), q_value in self.q_table.items():
                if s_key == next_state_key:
                    max_next_q = max(max_next_q, q_value)
            
            # Q-learning formula
            new_q = current_q + self.learning_rate * (
                experience.reward.value + self.discount_factor * max_next_q - current_q
            )
            
            self.q_table[(state_key, action_key)] = new_q
            
        except Exception as e:
            logging.error(f"Q-table update failed: {e}")
    
    def _train_neural_networks_from_experiences(self):
        """Train neural networks from experience data"""
        try:
            if len(self.experience_buffer) < 100:
                return  # Not enough data
            
            # Prepare training data
            X = []
            y = []
            
            for experience in list(self.experience_buffer)[-1000:]:  # Use last 1000 experiences
                # Extract features from state
                state_features = self._extract_state_features(experience.state)
                X.append(state_features)
                
                # Create target (reward-based)
                target = [0.0] * 10  # Assuming 10 possible actions
                action_index = self._action_to_index(experience.action)
                target[action_index] = experience.reward.value
                y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train neural networks
            for name, nn_data in self.neural_networks.items():
                if name == 'adaptation':
                    # Train adaptation network
                    nn_data['model'].fit(X, y, epochs=10, batch_size=32, verbose=0)
                    nn_data['trained'] = True
                    nn_data['last_trained'] = time.time()
                    self.metrics['neural_networks_trained'] += 1
            
            logging.info("Neural networks trained from experiences")
            
        except Exception as e:
            logging.error(f"Neural network training failed: {e}")
    
    def _extract_state_features(self, state: State) -> np.ndarray:
        """Extract features from state for neural network training"""
        try:
            features = [
                state.defense_accuracy,
                state.attack_success_rate,
                state.system_balance,
                state.performance_score,
                state.adaptation_level,
                state.threat_level,
                state.resource_utilization,
                state.learning_rate,
                state.evolution_rate,
                state.mutation_rate
            ]
            
            # Pad or truncate to fixed size
            while len(features) < 64:
                features.append(0.0)
            
            return np.array(features[:64])
            
        except Exception as e:
            logging.error(f"State feature extraction failed: {e}")
            return np.zeros(64)
    
    def _action_to_index(self, action: Action) -> int:
        """Convert action to index for neural network training"""
        action_mapping = {
            'adapt_defense': 0,
            'evolve_attack': 1,
            'balance_strategy': 2,
            'optimize_performance': 3,
            'increase_learning': 4,
            'decrease_learning': 5,
            'morph_system': 6,
            'analyze_threats': 7,
            'update_patterns': 8,
            'evolve_genome': 9
        }
        return action_mapping.get(action.action_type, 0)
    
    def _evolve_population_from_training(self):
        """Evolve genetic population based on training data"""
        try:
            if len(self.experience_buffer) < 50:
                return  # Not enough data
            
            # Calculate fitness based on recent experiences
            recent_experiences = list(self.experience_buffer)[-100:]
            avg_reward = np.mean([exp.reward.value for exp in recent_experiences])
            
            # Update population fitness
            for individual in self.population:
                individual.fitness = self._calculate_fitness_from_reward(individual, avg_reward)
            
            # Evolve population
            self._evolve_population()
            
            logging.info("Population evolved from training data")
            
        except Exception as e:
            logging.error(f"Population evolution failed: {e}")
    
    def _calculate_fitness_from_reward(self, individual: GeneticIndividual, avg_reward: float) -> float:
        """Calculate fitness based on training rewards"""
        try:
            # Base fitness from genes
            base_fitness = self._calculate_fitness(individual)
            
            # Adjust based on training performance
            reward_factor = max(0.1, min(2.0, avg_reward + 1.0))  # Normalize reward
            
            return base_fitness * reward_factor
            
        except Exception as e:
            logging.error(f"Fitness calculation failed: {e}")
            return 0.0
    
    def generate_training_scenarios(self, num_scenarios: int = 1000) -> List[Dict[str, Any]]:
        """Generate synthetic training scenarios for BALANCE controller"""
        scenarios = []
        
        # Scenario types
        scenario_types = {
            'high_threat': {
                'defense_accuracy': (0.3, 0.6),
                'attack_success_rate': (0.7, 0.9),
                'system_balance': (0.2, 0.4),
                'threat_level': (0.8, 1.0),
                'reward': (-0.5, 0.0)
            },
            'balanced': {
                'defense_accuracy': (0.5, 0.7),
                'attack_success_rate': (0.4, 0.6),
                'system_balance': (0.4, 0.6),
                'threat_level': (0.3, 0.7),
                'reward': (0.0, 0.5)
            },
            'defense_strong': {
                'defense_accuracy': (0.8, 1.0),
                'attack_success_rate': (0.1, 0.3),
                'system_balance': (0.7, 0.9),
                'threat_level': (0.1, 0.3),
                'reward': (0.5, 1.0)
            },
            'attack_strong': {
                'defense_accuracy': (0.2, 0.4),
                'attack_success_rate': (0.8, 1.0),
                'system_balance': (0.1, 0.3),
                'threat_level': (0.9, 1.0),
                'reward': (-1.0, -0.3)
            }
        }
        
        for _ in range(num_scenarios):
            scenario_type = random.choice(list(scenario_types.keys()))
            params = scenario_types[scenario_type]
            
            # Generate state
            state = {
                'defense_accuracy': random.uniform(*params['defense_accuracy']),
                'attack_success_rate': random.uniform(*params['attack_success_rate']),
                'system_balance': random.uniform(*params['system_balance']),
                'performance_score': random.uniform(0.0, 1.0),
                'adaptation_level': random.uniform(0.0, 1.0),
                'threat_level': random.uniform(*params['threat_level']),
                'resource_utilization': random.uniform(0.0, 1.0),
                'learning_rate': random.uniform(0.001, 0.1),
                'evolution_rate': random.uniform(0.01, 0.5),
                'mutation_rate': random.uniform(0.01, 0.3)
            }
            
            # Generate action
            action_types = ['adapt_defense', 'evolve_attack', 'balance_strategy', 'optimize_performance']
            action = {
                'action_type': random.choice(action_types),
                'parameters': {
                    'intensity': random.uniform(0.1, 1.0),
                    'duration': random.uniform(1.0, 10.0),
                    'target': random.choice(['defense', 'attack', 'balance'])
                }
            }
            
            # Generate reward
            reward = random.uniform(*params['reward'])
            
            # Generate next state (evolved from current state)
            next_state = state.copy()
            next_state['defense_accuracy'] = max(0.0, min(1.0, next_state['defense_accuracy'] + random.uniform(-0.1, 0.1)))
            next_state['attack_success_rate'] = max(0.0, min(1.0, next_state['attack_success_rate'] + random.uniform(-0.1, 0.1)))
            next_state['system_balance'] = max(0.0, min(1.0, next_state['system_balance'] + random.uniform(-0.1, 0.1)))
            
            scenario = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'scenario_type': scenario_type
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def evaluate_balance_performance(self) -> Dict[str, Any]:
        """Evaluate BALANCE controller performance"""
        try:
            # Calculate performance metrics
            total_actions = self.metrics['total_actions']
            successful_adaptations = self.metrics['successful_adaptations']
            failed_adaptations = self.metrics['failed_adaptations']
            
            if total_actions == 0:
                return {"error": "No actions performed"}
            
            success_rate = successful_adaptations / total_actions
            failure_rate = failed_adaptations / total_actions
            
            # Calculate average reward
            if self.reward_history:
                avg_reward = np.mean([reward.value for reward in self.reward_history[-100:]])
            else:
                avg_reward = 0.0
            
            # Q-table analysis
            q_values = list(self.q_table.values())
            avg_q_value = np.mean(q_values) if q_values else 0.0
            max_q_value = np.max(q_values) if q_values else 0.0
            
            # Population analysis
            if self.population:
                fitness_scores = [individual.fitness for individual in self.population]
                avg_fitness = np.mean(fitness_scores)
                max_fitness = np.max(fitness_scores)
                best_individual = max(self.population, key=lambda x: x.fitness)
            else:
                avg_fitness = 0.0
                max_fitness = 0.0
                best_individual = None
            
            # Neural network analysis
            nn_performance = {}
            for name, nn_data in self.neural_networks.items():
                nn_performance[name] = {
                    'trained': nn_data['trained'],
                    'accuracy': nn_data['accuracy'],
                    'last_trained': nn_data['last_trained']
                }
            
            performance = {
                'total_actions': total_actions,
                'success_rate': success_rate,
                'failure_rate': failure_rate,
                'average_reward': avg_reward,
                'q_table_size': len(self.q_table),
                'average_q_value': avg_q_value,
                'max_q_value': max_q_value,
                'population_size': len(self.population),
                'average_fitness': avg_fitness,
                'max_fitness': max_fitness,
                'best_individual_genes': best_individual.genes if best_individual else {},
                'neural_networks': nn_performance,
                'generation': self.generation,
                'threat_patterns_learned': self.metrics['threat_patterns_learned'],
                'system_morphs': self.metrics['system_morphs']
            }
            
            return performance
            
        except Exception as e:
            logging.error(f"Balance performance evaluation failed: {e}")
            return {"error": str(e)}
    
    def shutdown(self):
        """Shutdown the BALANCE Controller"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=5)
        
        # Save final state
        self.save_state()
        logging.info("Advanced BALANCE Controller shutdown complete")
