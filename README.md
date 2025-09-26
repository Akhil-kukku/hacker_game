# 🛡️ Self-Morphing AI Cybersecurity Engine v2.0

A comprehensive cybersecurity simulation platform featuring three AI-powered components that continuously evolve and adapt through machine learning, genetic algorithms, and reinforcement learning.

## 🎯 Overview

The Self-Morphing AI Cybersecurity Engine simulates a continuous attack-defense loop with three main components:

### 🛡️ ORDER (Defense Engine)
- **Isolation Forest** anomaly detection for network traffic analysis
- **Adaptive learning** with model mutation capabilities
- **Real-time flow processing** with batch optimization
- **Attack signature generation** and pattern recognition
- **Performance metrics** and accuracy tracking

### 🎯 CHAOS (Offensive Engine)
- **20+ attack types** including DDoS, SQL injection, XSS, brute force, and more
- **Evolutionary attack patterns** that adapt based on defense effectiveness
- **Stealth mechanisms** and detection avoidance
- **Adaptive payload generation** with varying complexity levels
- **Performance tracking** and success rate optimization

### ⚖️ BALANCE (Controller)
- **Reinforcement Learning** with Q-learning for optimal decision making
- **Genetic Algorithms** for parameter optimization
- **Real-time orchestration** of ORDER and CHAOS components
- **System balance optimization** and performance monitoring
- **Adaptive strategy evolution** based on outcomes

## 🚀 Features

### Core Capabilities
- **Continuous Simulation**: Automated attack-defense cycles with real-time adaptation
- **AI-Powered Evolution**: All components learn and adapt from interactions
- **Comprehensive Monitoring**: Real-time metrics, charts, and performance tracking
- **Interactive Dashboard**: Streamlit-based visualization and control interface
- **RESTful API**: Complete API for integration and automation
- **WebSocket Support**: Real-time updates and monitoring

### Advanced Features
- **Batch Processing**: Optimized data handling with chunk loading
- **Attack-Response Tracking**: Detailed interaction analysis
- **Model Persistence**: Save and load trained models and system state
- **Performance Optimization**: Automatic system tuning and optimization
- **Multi-threaded Architecture**: Concurrent processing for high performance

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Python Dependencies
```
fastapi==0.104.1
uvicorn==0.24.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
deap==1.4.1
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0
streamlit==1.28.1
requests==2.31.0
pydantic==2.4.2
python-dotenv==1.0.0
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd hacker_game
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Create Required Directories
```bash
mkdir -p data models logs
```

## 🚀 Quick Start

### Option 1: Run the Complete System
```bash
# Start the API server (includes all components)
python api_server.py
```

### Option 2: Run Individual Components
```bash
# Start the main engine
python main_engine.py

# Start the API server separately
python api_server.py

# Start the dashboard
streamlit run dashboard.py
```

### Option 3: Development Mode
```bash
# Run with auto-reload
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Run dashboard with auto-reload
streamlit run dashboard.py --server.port 8501
```

## 📊 Dashboard Access

Once running, access the interactive dashboard at:
- **Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 🔧 Configuration

### Engine Configuration
The system can be configured through the API or by modifying the default configurations in each component:

```python
# Example configuration
config = {
    'simulation_interval': 10.0,  # seconds
    'batch_size': 100,
    'auto_optimization': True,
    'performance_threshold': 0.7
}
```

### Component-Specific Settings

#### ORDER Engine
- `contamination`: Anomaly detection sensitivity (0.1)
- `n_estimators`: Number of isolation forest trees (100)
- `training_threshold`: Minimum samples for training (10000)
- `mutation_threshold`: Accuracy threshold for model mutation (0.8)

#### CHAOS Engine
- `max_concurrent_attacks`: Maximum simultaneous attacks (5)
- `stealth_threshold`: Stealth success threshold (0.7)
- `adaptation_threshold`: Success rate threshold for adaptation (0.3)
- `aggression_level`: Attack intensity (1-10)

#### BALANCE Controller
- `learning_rate`: Q-learning rate (0.1)
- `population_size`: Genetic algorithm population size (50)
- `control_interval`: Decision interval in seconds (5.0)
- `epsilon`: Exploration rate for RL (0.3)

## 📈 API Endpoints

### System Management
- `GET /` - System information
- `GET /health` - Health check
- `GET /status` - Comprehensive system status
- `POST /config` - Update configuration
- `POST /optimize` - Trigger optimization
- `POST /save` - Save system state
- `POST /load` - Load system state

### ORDER Engine (Defense)
- `GET /order/status` - Defense engine status
- `GET /order/signatures` - Attack signatures
- `POST /flows` - Process network flows

### CHAOS Engine (Offense)
- `GET /chaos/status` - Attack engine status
- `GET /chaos/results` - Attack results
- `GET /chaos/patterns` - Attack patterns
- `POST /attacks` - Launch attacks
- `POST /chaos/aggression` - Set aggression level
- `POST /chaos/stealth` - Set stealth mode

### BALANCE Controller
- `GET /balance/status` - Controller status
- `GET /balance/actions` - Action history
- `GET /balance/rewards` - Reward history

### Data and Analytics
- `GET /simulations` - Simulation results
- `GET /tracking` - Attack-response tracking
- `WebSocket /ws` - Real-time updates

## 🎮 Usage Examples

### Launch an Attack
```python
import requests

# Launch a DDoS attack
attack_data = [{
    "attack_type": "DDoS",
    "target_ip": "192.168.1.1",
    "target_port": 80
}]

response = requests.post("http://localhost:8000/attacks", json=attack_data)
print(response.json())
```

### Process Network Flows
```python
# Process network flows for analysis
flows_data = [{
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.0.1",
    "src_port": 12345,
    "dst_port": 80,
    "protocol": "TCP",
    "packet_count": 100,
    "byte_count": 1024,
    "duration": 1.5,
    "flags": "SYN"
}]

response = requests.post("http://localhost:8000/flows", json=flows_data)
print(response.json())
```

### Get System Status
```python
# Get comprehensive system status
response = requests.get("http://localhost:8000/status")
status = response.json()

print(f"System Balance: {status['performance_metrics']['system_balance_score']}")
print(f"Total Simulations: {status['performance_metrics']['total_simulations']}")
```

## 📊 Monitoring and Analytics

### Dashboard Features
- **Real-time Metrics**: Live system performance indicators
- **Interactive Charts**: Performance trends and comparisons
- **Control Panel**: Direct system control and configuration
- **Attack Analysis**: Detailed attack signature and result analysis
- **System Optimization**: One-click optimization triggers

### Key Metrics Tracked
- **System Balance**: Overall attack-defense equilibrium
- **Defense Accuracy**: ORDER engine detection effectiveness
- **Attack Success Rate**: CHAOS engine success metrics
- **Learning Progress**: BALANCE controller evolution
- **Performance Trends**: Historical performance analysis

## 🔍 Troubleshooting

### Common Issues

#### API Connection Errors
```bash
# Check if API server is running
curl http://localhost:8000/health

# Check logs
tail -f backend/main_engine.log
```

#### Dashboard Connection Issues
```bash
# Verify API server is running
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"

# Check Streamlit logs
streamlit run dashboard.py --logger.level debug
```

#### Performance Issues
- Increase `batch_size` for better throughput
- Adjust `simulation_interval` for different update frequencies
- Monitor memory usage and adjust population sizes

### Log Files
- `main_engine.log` - Main engine logs
- `order_engine.log` - Defense engine logs
- `chaos_engine.log` - Attack engine logs
- `balance_controller.log` - Controller logs

## 🔬 Advanced Usage

### Custom Attack Types
```python
from chaos_engine import AttackType

# Add custom attack type
class CustomAttack(AttackType):
    CUSTOM = "Custom Attack"

# Implement custom payload generation
def generate_custom_payload(self, size: int) -> bytes:
    # Custom payload logic
    return b"custom_payload"
```

### Custom Defense Strategies
```python
from order_engine import OrderEngine

# Extend ORDER engine with custom features
class CustomOrderEngine(OrderEngine):
    def custom_feature_extraction(self, flow):
        # Custom feature extraction logic
        return custom_features
```

### Integration with External Systems
```python
# WebSocket integration for real-time monitoring
import websockets
import asyncio

async def monitor_system():
    async with websockets.connect('ws://localhost:8000/ws') as websocket:
        while True:
            status = await websocket.recv()
            print(f"System Status: {status}")

asyncio.run(monitor_system())
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

## 🔮 Future Enhancements

### Planned Features
- **XAI Integration**: Explainable AI tools for ORDER engine decisions
- **Docker Support**: Containerized deployment
- **CI/CD Pipeline**: Automated testing and deployment
- **Cloud Integration**: AWS/Azure deployment options
- **Advanced ML Models**: Deep learning and neural networks
- **Real Network Integration**: Live network monitoring capabilities

### Roadmap
- **Q1 2024**: XAI tools and advanced visualization
- **Q2 2024**: Docker and cloud deployment
- **Q3 2024**: Advanced ML models and real network integration
- **Q4 2024**: Enterprise features and scalability improvements

---

**🛡️ Self-Morphing AI Cybersecurity Engine v2.0** - Where AI meets cybersecurity in an ever-evolving battle of wits.
