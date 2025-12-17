# Contributing to Pose-Based Fitness Coach

First off, thank you for considering contributing to the Pose-Based Fitness Coach! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our commitment to maintaining a welcoming and inclusive environment. Please be respectful and considerate in all interactions.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Webcam (for testing)
- Basic understanding of pose detection concepts

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/pose-fitness-coach.git
   cd pose-fitness-coach
   ```

3. Add the original repository as upstream:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/pose-fitness-coach.git
   ```

## 🤝 How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs **actual behavior**
- **Screenshots/videos** if applicable
- **Environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Use case** - Why is this feature needed?
- **Proposed solution** - How would it work?
- **Alternatives considered** - Other approaches you've thought of

### Adding New Exercises

See [docs/adding_exercises.md](docs/adding_exercises.md) for a detailed guide.

Quick overview:
1. Create a new tracker class in `src/exercises/`
2. Inherit from `BaseExerciseTracker`
3. Implement required methods
4. Add tests in `tests/`
5. Update documentation

## 💻 Development Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## 📝 Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) guidelines
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Documentation

- All public functions/classes must have docstrings
- Use Google-style docstrings:
  ```python
  def function_name(param1: int, param2: str) -> bool:
      """
      Brief description of function.
      
      Args:
          param1: Description of param1
          param2: Description of param2
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: When something is wrong
      """
  ```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs where appropriate

Example:
```
Add lateral raise exercise tracker

- Implement LateralRaiseTracker class
- Add angle calculations for shoulder abduction
- Include form feedback for elbow position
- Add unit tests

Closes #123
```

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Test edge cases (missing landmarks, low visibility, etc.)

## 🔄 Pull Request Process

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them

3. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

4. **Update documentation** if needed

5. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request** on GitHub

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] PR description explains changes

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## 🏷️ Labels

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `enhancement` | New feature request |
| `documentation` | Documentation improvements |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |
| `exercise` | New exercise tracker |

## ❓ Questions?

Feel free to:
- Open an issue with the `question` label
- Start a discussion on GitHub Discussions
- Reach out to maintainers

---

Thank you for contributing! 🙏
