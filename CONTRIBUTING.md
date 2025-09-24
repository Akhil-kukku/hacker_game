# Contributing to Self-Morphing AI Cybersecurity Engine

Thank you for your interest in contributing to the Self-Morphing AI Cybersecurity Engine! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git
- Docker (optional, for containerized development)

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/hacker_game.git
   cd hacker_game
   ```
3. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
4. Run the system:
   ```bash
   python start_engine.py
   ```

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all functions and classes
- Keep functions focused and single-purpose

### Testing
- Write unit tests for new features
- Ensure all tests pass before submitting a PR
- Test on multiple Python versions (3.9, 3.10, 3.11, 3.12)

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
- **ORDER Engine**: Defense system using Isolation Forest
- **CHAOS Engine**: Offensive system with adaptive attacks
- **BALANCE Controller**: RL + GA orchestration system

### File Structure
```
backend/
├── order_engine.py      # Defense component
├── chaos_engine.py      # Attack component
├── balance_controller.py # Control component
├── main_engine.py       # Main orchestrator
├── api_server.py        # FastAPI server
├── dashboard.py         # Streamlit dashboard
└── requirements.txt     # Dependencies
```

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Error message and stack trace
- Steps to reproduce
- Expected vs actual behavior

## 💡 Feature Requests

For feature requests:
- Describe the feature clearly
- Explain the use case
- Consider implementation complexity
- Discuss potential impact on existing components

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
