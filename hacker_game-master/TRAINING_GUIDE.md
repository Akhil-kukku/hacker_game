# 🎯 Training Guide: Developing Base Defense and Attack Capabilities

This guide explains how to train the Self-Morphing AI Cybersecurity Engine with known attacks to develop a strong baseline defense and attack capability.

## 🚀 Quick Start Training

### 1. **Start the System**
```bash
# Start the API server
cd backend
python api_server.py

# In another terminal, start the dashboard
python dashboard.py
```

### 2. **Run Comprehensive Training**
```bash
# Run the automated training script
python train_models.py
```

This will train all three engines with 2000 samples each, providing a solid foundation for cybersecurity operations.

## 📚 Detailed Training Methods

### 🛡️ ORDER Engine (Defense) Training

The ORDER engine learns to detect and respond to known attack patterns:

#### **API Training**
```python
import requests

# Train ORDER engine with 1000 samples
response = requests.post("http://localhost:8000/order/train?num_samples=1000")
result = response.json()
print(f"Training result: {result}")
```

#### **Custom Training Data**
```python
# Create custom training data
training_data = [
    {
        "attack_type": "ddos",
        "flow": {
            "packet_count": 5000,
            "byte_count": 500000,
            "duration": 0.5,
            "src_port": 12345,
            "dst_port": 80,
            "protocol": 6,  # TCP
            "flags": 2,     # SYN
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1"
        },
        "attack_features": {
            "payload_size": 500,
            "request_rate": 500,
            "response_time": 0.1,
            "error_rate": 0.1,
            "connection_count": 50
        },
        "behavior": {
            "stealth_level": 2,
            "aggression": 9,
            "complexity": 3,
            "persistence": 5
        }
    }
    # ... more samples
]

# Train with custom data
response = requests.post("http://localhost:8000/order/train", json=training_data)
```

#### **Attack Types Supported**
- **DDoS**: High packet count, short duration, low stealth
- **SQL Injection**: Medium complexity, high stealth, specific payloads
- **XSS**: Web-based attacks, medium stealth
- **Brute Force**: High frequency, low stealth, authentication attacks
- **Normal Traffic**: Baseline for comparison

### 🎯 CHAOS Engine (Attack) Training

The CHAOS engine learns effective attack patterns and counterattack strategies:

#### **API Training**
```python
# Train CHAOS engine with 1000 samples
response = requests.post("http://localhost:8000/chaos/train?num_samples=1000")
result = response.json()
print(f"Training result: {result}")
```

#### **Custom Attack Training**
```python
# Create custom attack training data
attack_data = [
    {
        "attack_type": "ddos",
        "success_rate": 0.8,
        "damage_dealt": 8,
        "stealth_maintained": False,
        "response_time": 0.2,
        "complexity": 3
    }
    # ... more samples
]

# Train with custom data
response = requests.post("http://localhost:8000/chaos/train", json=attack_data)
```

#### **Attack Pattern Evolution**
The CHAOS engine automatically evolves attack patterns based on:
- Success rates
- Damage potential
- Stealth maintenance
- Response times
- Complexity levels

### ⚖️ BALANCE Controller (Orchestration) Training

The BALANCE controller learns to orchestrate defense and attack strategies:

#### **API Training**
```python
# Train BALANCE controller with 1000 scenarios
response = requests.post("http://localhost:8000/balance/train?num_scenarios=1000")
result = response.json()
print(f"Training result: {result}")
```

#### **Custom Scenario Training**
```python
# Create custom training scenarios
scenarios = [
    {
        "state": {
            "defense_accuracy": 0.7,
            "attack_success_rate": 0.3,
            "system_balance": 0.6,
            "threat_level": 0.4,
            "performance_score": 0.8
        },
        "action": {
            "action_type": "adapt_defense",
            "parameters": {
                "intensity": 0.5,
                "duration": 5.0,
                "target": "defense"
            }
        },
        "reward": 0.8,
        "next_state": {
            "defense_accuracy": 0.8,
            "attack_success_rate": 0.2,
            "system_balance": 0.7,
            "threat_level": 0.3,
            "performance_score": 0.9
        }
    }
    # ... more scenarios
]

# Train with custom scenarios
response = requests.post("http://localhost:8000/balance/train", json=scenarios)
```

## 🔧 Advanced Training Techniques

### 1. **Incremental Training**
Train the system gradually with increasing complexity:

```python
# Phase 1: Basic attacks
trainer.train_all_engines(order_samples=500, chaos_samples=500, balance_scenarios=500)

# Phase 2: Intermediate attacks
trainer.train_all_engines(order_samples=1000, chaos_samples=1000, balance_scenarios=1000)

# Phase 3: Advanced attacks
trainer.train_all_engines(order_samples=2000, chaos_samples=2000, balance_scenarios=2000)
```

### 2. **Real-World Data Integration**
Integrate real-world attack data:

```python
# Load real attack data from files
import json

with open("real_attacks.json", "r") as f:
    real_attacks = json.load(f)

# Train with real data
trainer.train_with_custom_data(order_data=real_attacks)
```

### 3. **Performance Monitoring**
Monitor training progress:

```python
# Check ORDER performance
order_perf = requests.get("http://localhost:8000/order/performance").json()
print(f"ORDER accuracy: {order_perf['accuracy']:.3f}")

# Check CHAOS performance
chaos_perf = requests.get("http://localhost:8000/chaos/performance").json()
print(f"CHAOS success rate: {chaos_perf['success_rate']:.3f}")

# Check BALANCE performance
balance_perf = requests.get("http://localhost:8000/balance/performance").json()
print(f"BALANCE average reward: {balance_perf['average_reward']:.3f}")
```

## 📊 Training Data Formats

### ORDER Engine Training Data Format
```json
{
    "attack_type": "ddos|sql_injection|xss|brute_force|normal",
    "flow": {
        "packet_count": 1000,
        "byte_count": 100000,
        "duration": 1.0,
        "src_port": 12345,
        "dst_port": 80,
        "protocol": 6,
        "flags": 2,
        "src_ip": "192.168.1.100",
        "dst_ip": "10.0.0.1",
        "timestamp": 1640995200.0
    },
    "attack_features": {
        "payload_size": 500,
        "request_rate": 100,
        "response_time": 0.5,
        "error_rate": 0.1,
        "connection_count": 10
    },
    "behavior": {
        "stealth_level": 5,
        "aggression": 7,
        "complexity": 4,
        "persistence": 3
    }
}
```

### CHAOS Engine Training Data Format
```json
{
    "attack_type": "ddos|sql_injection|xss|brute_force|phishing",
    "success_rate": 0.7,
    "damage_dealt": 6,
    "stealth_maintained": true,
    "response_time": 0.5,
    "complexity": 5,
    "timestamp": 1640995200.0
}
```

### BALANCE Controller Training Data Format
```json
{
    "state": {
        "defense_accuracy": 0.7,
        "attack_success_rate": 0.3,
        "system_balance": 0.6,
        "performance_score": 0.8,
        "adaptation_level": 0.5,
        "threat_level": 0.4,
        "resource_utilization": 0.6,
        "learning_rate": 0.01,
        "evolution_rate": 0.1,
        "mutation_rate": 0.05
    },
    "action": {
        "action_type": "adapt_defense|evolve_attack|balance_strategy|optimize_performance",
        "parameters": {
            "intensity": 0.5,
            "duration": 5.0,
            "target": "defense|attack|balance"
        }
    },
    "reward": 0.8,
    "next_state": {
        "defense_accuracy": 0.8,
        "attack_success_rate": 0.2,
        "system_balance": 0.7,
        "performance_score": 0.9,
        "adaptation_level": 0.6,
        "threat_level": 0.3,
        "resource_utilization": 0.5,
        "learning_rate": 0.01,
        "evolution_rate": 0.1,
        "mutation_rate": 0.05
    },
    "scenario_type": "high_threat|balanced|defense_strong|attack_strong"
}
```

## 🎯 Training Best Practices

### 1. **Start with Synthetic Data**
- Use the built-in data generators for initial training
- Ensure balanced representation of attack types
- Include normal traffic patterns

### 2. **Gradual Complexity Increase**
- Begin with simple attacks (DDoS, brute force)
- Progress to complex attacks (SQL injection, XSS)
- Add advanced persistent threats (APTs)

### 3. **Performance Validation**
- Regularly evaluate model performance
- Monitor accuracy, precision, recall, and F1 scores
- Adjust training parameters based on results

### 4. **Real-World Integration**
- Gradually introduce real-world attack data
- Validate against known attack signatures
- Test with live network traffic

### 5. **Continuous Learning**
- Implement online learning for new attack patterns
- Update models with new threat intelligence
- Maintain model performance over time

## 🔍 Troubleshooting

### Common Issues

1. **Low Accuracy**
   - Increase training samples
   - Check feature extraction
   - Verify data quality

2. **Overfitting**
   - Reduce model complexity
   - Increase regularization
   - Use more diverse training data

3. **Poor Performance**
   - Check API server status
   - Verify training data format
   - Monitor system resources

### Performance Targets

- **ORDER Engine**: Accuracy > 0.85, F1 Score > 0.80
- **CHAOS Engine**: Success Rate > 0.70, Stealth Rate > 0.60
- **BALANCE Controller**: Average Reward > 0.50, Success Rate > 0.80

## 📈 Monitoring and Evaluation

### Real-Time Monitoring
```python
# Monitor training progress
while training:
    order_perf = requests.get("http://localhost:8000/order/performance").json()
    chaos_perf = requests.get("http://localhost:8000/chaos/performance").json()
    balance_perf = requests.get("http://localhost:8000/balance/performance").json()
    
    print(f"ORDER Accuracy: {order_perf.get('accuracy', 0):.3f}")
    print(f"CHAOS Success: {chaos_perf.get('success_rate', 0):.3f}")
    print(f"BALANCE Reward: {balance_perf.get('average_reward', 0):.3f}")
    
    time.sleep(10)
```

### Training Reports
The system generates comprehensive training reports including:
- Model performance metrics
- Training data statistics
- System readiness assessment
- Recommendations for improvement

## 🚀 Next Steps

After successful training:

1. **Deploy in Production**: Use the trained models in real-world environments
2. **Monitor Performance**: Continuously monitor and evaluate system performance
3. **Update Models**: Regularly retrain with new attack patterns
4. **Expand Capabilities**: Add new attack types and defense mechanisms
5. **Integrate Intelligence**: Connect with threat intelligence feeds

The trained system will now have a solid foundation for defending against known attacks while continuously learning and adapting to new threats through its self-morphing AI capabilities.

