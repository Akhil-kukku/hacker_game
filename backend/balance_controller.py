"""
BALANCE Controller - Orchestration System
Self-Morphing AI Cybersecurity Engine - Control Component
"""

import numpy as np
import random
import time
import threading
import queue
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import copy
from collections import deque
import pickle
import os

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
    BALANCE Controller using Reinforcement Learning and Genetic Algorithms
    to orchestrate ORDER and CHAOS engines
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # RL components
        self.q_table = {}
        self.experience_buffer = deque(maxlen=self.config['experience_buffer_size'])
        self.epsilon = self.config['initial_epsilon']
        self.learning_rate = self.config['learning_rate']
        self.discount_factor = self.config['discount_factor']
        
        # Genetic algorithm components
        self.population = []
        self.population_size = self.config['population_size']
        self.generation = 0
        self.best_individual = None
        
        # System state tracking
        self.current_state = None
        self.action_history = []
        self.reward_history = []
        self.performance_history = []
        
        # Control parameters
        self.defense_weight = 0.5
        self.attack_weight = 0.5
        self.balance_threshold = 0.6
        
        # Performance metrics
        self.metrics = {
            'total_actions': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'average_reward': 0.0,
            'best_fitness': 0.0,
            'generation_count': 0,
            'last_optimization': None
        }
        
        # Initialize components
        self._initialize_genetic_population()
        self._initialize_q_table()
        
        # Start control loop
        self.running = False
        self.control_thread = None
        
        logging.info("BALANCE Controller initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for BALANCE Controller"""
        return {
            'experience_buffer_size': 1000,
            'initial_epsilon': 0.3,
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'population_size': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_size': 5,
            'generation_limit': 100,
            'fitness_threshold': 0.8,
            'control_interval': 5.0,  # seconds
            'optimization_threshold': 0.7,
            'save_path': 'models/balance_controller.pkl'
        }
    
    def _initialize_genetic_population(self):
        """Initialize the genetic algorithm population"""
        for _ in range(self.population_size):
            individual = GeneticIndividual({
                'defense_adaptation_rate': random.uniform(0.1, 0.9),
                'attack_evolution_rate': random.uniform(0.1, 0.9),
                'balance_sensitivity': random.uniform(0.1, 0.9),
                'mutation_threshold': random.uniform(0.3, 0.8),
                'learning_rate': random.uniform(0.05, 0.2),
                'aggression_level': random.randint(1, 10),
                'stealth_preference': random.uniform(0.1, 0.9),
                'adaptation_frequency': random.randint(1, 10)
            })
            self.population.append(individual)
        
        logging.info(f"Genetic population initialized with {self.population_size} individuals")
    
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
        # Try to read real metrics if available via API; fallback to simulated
        defense_accuracy = random.uniform(0.6, 0.9)
        attack_success_rate = random.uniform(0.3, 0.7)
        system_balance = (defense_accuracy + (1 - attack_success_rate)) / 2
        try:
            import requests
            r = requests.get('http://localhost:8000/status', timeout=1.0)
            if r.status_code == 200:
                s = r.json()
                order = s.get('order_engine', {})
                chaos = s.get('chaos_engine', {})
                perf = order.get('performance_metrics', {})
                # Approximate accuracy as 1 - anomalies/total
                total = perf.get('total_flows_processed', 0) or 0
                anomalies = perf.get('anomalies_detected', 0) or 0
                if total > 0:
                    defense_accuracy = max(0.0, min(1.0, 1.0 - anomalies / total))
                c_total = chaos.get('total_attacks', 0) or 0
                c_success = chaos.get('successful_attacks', 0) or 0
                if c_total > 0:
                    attack_success_rate = max(0.0, min(1.0, c_success / c_total))
                system_balance = (defense_accuracy + (1 - attack_success_rate)) / 2
        except Exception:
            pass
        
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
    
    def shutdown(self):
        """Shutdown the BALANCE Controller"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=5)
        
        # Save final state
        self.save_state()
        logging.info("BALANCE Controller shutdown complete")
