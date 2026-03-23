# Contributing to AILS

> **Artificial Intelligence Learning System** — Created by Cherry Computer Ltd.

Thank you for considering contributing to AILS! This project thrives on community involvement, and we welcome contributions of all kinds — bug fixes, new features, documentation improvements, tests, and more.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [Ethical Contribution Guidelines](#ethical-contribution-guidelines)

---

## 📜 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- A GitHub account

### Fork & Clone

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/AILS.git
cd AILS

# 3. Add upstream remote
git remote add upstream https://github.com/CherryComputerLtd/AILS.git
```

---

## 🛠️ Development Setup

```bash
# Create a virtual environment
python -m venv ails-dev-env
source ails-dev-env/bin/activate  # Windows: ails-dev-env\Scripts\activate

# Install dependencies + dev tools
pip install -r requirements.txt
pip install -e ".[dev]"

# Verify everything works
pytest tests/ -v
```

---

## 🤝 How to Contribute

### Step-by-Step Workflow

1. **Sync with upstream**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make your changes**
   - Write clean, documented code
   - Add/update tests for your changes
   - Update documentation if needed

4. **Run tests & checks**
   ```bash
   # Run all tests
   pytest tests/ -v --cov=src

   # Check code formatting
   black src/ tests/

   # Lint
   flake8 src/ tests/ --max-line-length=100
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat(nlp): add multilingual tokenization support"
   ```

6. **Push and open a PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub.

---

## 🎨 Coding Standards

### Python Style Guide

- Follow **PEP 8** for all Python code
- Use **Black** for formatting (line length: 88)
- Use **type hints** for all function parameters and return values
- Write **docstrings** for all public classes and methods
- Use f-strings for string formatting

### Example of Good Code

```python
def clean_text(self, text: str,
               remove_stopwords: bool = True) -> str:
    """
    Clean and normalize a text string.

    Args:
        text: The input text to clean.
        remove_stopwords: Whether to remove common stopwords.

    Returns:
        The cleaned text string.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    # ... implementation
```

### Documentation Standards

- All public APIs must have docstrings
- Include parameter types and descriptions
- Document return values and exceptions
- Provide usage examples in docstrings

---

## 📝 Commit Messages

Use the **Conventional Commits** specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting changes |
| `refactor` | Code refactoring |
| `test` | Adding/updating tests |
| `chore` | Build, CI, or tooling changes |
| `perf` | Performance improvements |
| `security` | Security fixes |

### Examples

```
feat(nlp): add multilingual sentiment analysis
fix(database): resolve connection pool timeout issue
docs(readme): update installation instructions
test(ethics): add edge cases for disparate impact
```

---

## 🔀 Pull Request Process

1. **Ensure tests pass**: All CI checks must pass
2. **One feature per PR**: Keep PRs focused and small
3. **Update documentation**: Reflect your changes
4. **Write a clear description**: Explain what, why, and how
5. **Link issues**: Reference related issues with `Closes #123`
6. **Add a screenshot/demo** for UI changes

### PR Title Format

```
[feat] Add multilingual NLP support
[fix] Resolve database connection timeout
[docs] Improve deployment guide
```

### PR Description Template

Your PR should include:
- **Summary**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Changes**: List of changes made
- **Testing**: How was it tested?
- **Screenshots**: (if applicable)

---

## 🐛 Reporting Bugs

When reporting a bug, please include:

1. **Environment**: OS, Python version, AILS version
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Error output**: Full stack trace if available

Use the [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).

---

## 💡 Feature Requests

We love new ideas! When requesting a feature:

1. Check if it already exists or is planned
2. Explain the use case and motivation
3. Provide a code example of the desired API
4. Discuss potential implementation approaches

Use the [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md).

---

## ⚖️ Ethical Contribution Guidelines

AILS is built with an ethics-first approach. All contributions **must** adhere to:

### 1. Bias Awareness
- Test models for demographic bias
- Do not introduce training data that amplifies existing biases
- Document potential biases in new models

### 2. Privacy Protection
- Never include personally identifiable information (PII) in code or tests
- Use synthetic or anonymized data in examples
- Respect data minimization principles

### 3. Transparency
- Document model limitations clearly
- Explain decision-making in AI components
- Avoid "black box" implementations without explainability support

### 4. Responsible Scraping
- Respect `robots.txt` directives
- Implement rate limiting in all scrapers
- Do not scrape private or copyrighted data

### 5. Legal Compliance
- Ensure contributions comply with GDPR and CCPA
- Do not include code that violates terms of service
- Properly license all contributed code

---

## 🏆 Recognition

All contributors will be recognized in:
- The [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- The project's release notes
- The README acknowledgements section

---

*Thank you for helping make AILS better! — Cherry Computer Ltd.*
