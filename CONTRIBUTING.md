# Contributing to Self-Morphing AI Cybersecurity Engine

**Last Updated: November 2025**

Thank you for your interest in contributing to the Self-Morphing AI Cybersecurity Engine! This document provides guidelines and information for contributors.

## 🌐 **Project Context (2025)**

This project addresses critical cybersecurity challenges facing organizations in 2025:
- **Detection Crisis**: Traditional systems miss 71% of attacks ([Omdia Research](https://omdia.tech.informa.com))
- **Cost Impact**: $4.88M average breach cost ([IBM Security](https://www.ibm.com/security/data-breach))
- **Threat Volume**: 150+ actively exploited vulnerabilities in 2025 ([CISA](https://www.cisa.gov/known-exploited-vulnerabilities-catalog))
- **Response Time**: 207-day average breach detection time ([IBM](https://www.ibm.com/security/data-breach))

**Our Solution**: ML-based adaptive defense with 80%+ detection rate, <50ms processing, and real-time learning.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.14+** (tested on 3.14.0)
- **Git** (for version control)
- **Docker** (optional, for containerized development)
- **Basic ML knowledge** (scikit-learn, pandas)
- **Cybersecurity understanding** (MITRE ATT&CK, network flows)

### Development Setup
1. **Fork the repository**
   - Click "Fork" on GitHub: [github.com/Akhil-kukku/hacker_game](https://github.com/Akhil-kukku/hacker_game)

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/hacker_game.git
   cd hacker_game
   ```

3. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Generate training dataset** (if not present):
   ```bash
   python tools/generate_sample_dataset.py
   # Generates 12,399 samples in CSV Files/
   ```

5. **Run the system**:
   ```bash
   python start_engine.py
   # Backend: http://localhost:8000
   # Dashboard: http://localhost:8501
   ```

6. **Verify training**:
   ```bash
   # Check logs for:
   # "🎓 Found training dataset"
   # "✅ ORDER engine trained successfully"
   ```

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all functions and classes
- Keep functions focused and single-purpose

### Testing
- Write unit tests for new features (use pytest framework)
- Ensure all tests pass before submitting a PR
- Test on multiple Python versions (3.14+)
- **Test coverage target**: 80%+ for new code
- Run tests: `pytest backend/tests/ -v`
- Performance testing: Verify <50ms flow processing time

### Performance Benchmarks
New contributions should maintain or improve:
- **Detection Rate**: ≥80%
- **False Positive Rate**: <25%
- **Processing Speed**: <50ms per flow
- **Memory Usage**: <512MB for 10K flow cache
- **Training Time**: <5 seconds for 10K samples

### Commit Messages
Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

### Pull Request Process
1. Create a feature branch from `main`
2. Make your changes
3. Add tests if applicable
4. Update documentation
5. Run linting: `flake8 backend/ --max-line-length=120`
6. Submit a PR with a clear description

## 🏗️ Architecture Overview

### Core Components
- **ORDER Engine** (`order_engine.py`): Defense system using Isolation Forest
  - Real data training on UNSW-NB15 schema (12,399 samples)
  - 80%+ detection rate, <50ms processing
  - Adaptive mutation every 10-15 false negatives
  
- **CHAOS Engine** (`chaos_engine.py`): Offensive system with adaptive attacks
  - 20 attack types (DDoS, SQL Injection, XSS, etc.)
  - Detection rates: DDoS 85%, Brute Force 88%, Zero-Day 72%
  - Simulates 2025 threat actors: Lazarus, MuddyWater, BlueNoroff
  
- **BALANCE Controller** (`balance_controller.py`): RL + GA orchestration
  - Q-Learning with confusion matrix tracking (TP/FP/TN/FN)
  - Genetic Algorithm with 50-individual population
  - 8-12 feedback loops per simulation batch

### File Structure
```
backend/
├── order_engine.py      # Defense ML model (IsolationForest)
├── chaos_engine.py      # Attack simulator (20 types)
├── balance_controller.py # RL+GA controller
├── main_engine.py       # Main orchestrator with feedback loop
├── api_server.py        # FastAPI REST API (port 8000)
├── dashboard.py         # Streamlit UI (port 8501)
├── game_logic.py        # Legacy game components
├── requirements.txt     # Python dependencies
└── data/
    ├── attack_signatures.json  # Learned patterns
    └── system_state.json       # Saved state
    
CSV Files/
├── training_data.csv    # 9,920 samples (80%)
└── test_data.csv        # 2,479 samples (20%)

tools/
└── generate_sample_dataset.py  # Dataset generator
```

### Data Flow
1. **Network Flows** → ORDER Engine (anomaly detection)
2. **Detected Anomalies** → CHAOS Engine (adaptive evolution)
3. **Attack-Defense Results** → BALANCE Controller (strategy optimization)
4. **Feedback Loop** → ORDER Engine retraining (continuous learning)

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Error message and stack trace
- Steps to reproduce
- Expected vs actual behavior

## 💡 Feature Requests

For feature requests:
- Describe the feature clearly with use case
- Explain the cybersecurity problem it solves
- Reference 2025 threat landscape if applicable ([CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog), [MITRE ATT&CK](https://attack.mitre.org/))
- Consider implementation complexity and performance impact
- Discuss potential impact on existing components (ORDER/CHAOS/BALANCE)

### Priority Areas for Contributions
1. **Enhanced Detection** (Q1-Q2 2026):
   - Deep learning models (LSTM, GNN)
   - UNSW-NB15 full dataset integration (2.5M samples)
   - Zero-day detection improvements (target 95%+)
   
2. **Advanced Analytics** (Q3 2026):
   - XAI integration (SHAP/LIME)
   - Real-time dashboards with drill-down
   - Compliance reporting (GDPR, SOC 2, ISO 27001)
   
3. **Enterprise Features** (Q4 2026):
   - Kubernetes deployment manifests
   - SIEM connectors (Splunk, QRadar, Sentinel)
   - EDR/XDR integrations

See full roadmap in [README.md](README.md#future-enhancements)

---

## 📚 **Resources for Contributors**

### Cybersecurity References:
- [MITRE ATT&CK Framework](https://attack.mitre.org/) - Threat actor tactics
- [CISA Known Exploited Vulnerabilities](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) - Active threats
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework) - Security standards
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - Application security risks

### Machine Learning Resources:
- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) - Training data
- [Scikit-learn Documentation](https://scikit-learn.org/) - ML library
- [IsolationForest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) - Algorithm theory

### Project Documentation:
- [CODE_ANALYSIS.md](CODE_ANALYSIS.md) - Complete system analysis
- [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) - Recent fixes log
- [PRESENTATION_CONTENT.md](PRESENTATION_CONTENT.md) - Project overview

---

## 🔒 Security

- Report security vulnerabilities privately
- Do not include sensitive data in issues or PRs
- Follow responsible disclosure practices

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## 🤝 Community

- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and best practices
- Participate in discussions and code reviews

Thank you for contributing to the Self-Morphing AI Cybersecurity Engine! 🛡️
