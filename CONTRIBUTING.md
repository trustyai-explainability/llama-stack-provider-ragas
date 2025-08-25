# Contributing Guide

Thank you for your interest in contributing to the RAGAS provider! This guide will help you get started.

## Quick Start

- Review [README.md](README.md) for installation and usage instructions.
- Make sure you can run the tests (including the integration tests).

## Development Workflow

### 1. Fork and Clone
- Fork the repository
- Clone your fork: `git clone <your-fork-url>`
- Add upstream: `git remote add upstream <original-repo-url>`

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Write your code
- Add tests for new functionality
- Ensure existing tests pass

### 4. Run Quality Checks
```bash
# Run all checks
pre-commit run --all-files

# Or run individual checks
pre-commit run ruff      # Formatting and linting
pre-commit run mypy      # Type checking
pre-commit run pytest    # Unit tests
```

### 5. Commit and Push
```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 6. Submit Pull Request
- Create PR from your branch to main
- Ensure CI checks pass
- Request review from maintainers

## Code Quality Standards

### Pre-commit Hooks
The following checks run automatically on every commit:

- **Code formatting**: ruff format
- **Linting**: ruff with auto-fix
- **Type checking**: mypy
- **Unit tests**: pytest (excluding integration tests)

### Test Guidelines
- **Unit tests**: Required for all new functionality
- **Integration tests**: Mark with `@pytest.mark.integration_test`
- **Test coverage**: Aim for high coverage on new code

### Code Style
- Follow PEP 8 (enforced by ruff)
- Use type hints where appropriate
- Write clear docstrings for public functions

## Available Commands

```bash
# Setup
pre-commit install              # Install git hooks

# Quality checks
pre-commit run --all-files      # Run all checks
pre-commit run ruff             # Formatting and linting
pre-commit run mypy             # Type checking
pre-commit run pytest           # Unit tests

# Testing
uv run pytest tests/ -v                    # All tests
uv run pytest tests/ -m "not integration_test"  # Unit tests only
uv run pytest tests/ --cov=src            # With coverage
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Pre-commit not running | `pre-commit install` |
| Dependencies out of sync | `uv sync --extra dev` |
| Type checking errors | Check imports and type hints |
| Test failures | Verify test markers and dependencies |
| Hook failures | `pre-commit clean && pre-commit autoupdate` |

### Getting Help

- Check existing issues and PRs
- Ask questions in discussions
- Review the codebase for examples

## Configuration Files

- **`.pre-commit-config.yaml`**: Git hooks configuration
- **`pyproject.toml`**: Project settings and tool configuration

## IDE Setup

### VS Code
Install: Python, Ruff, MyPy Type Checker, Pytest

### PyCharm
Enable: MyPy integration, pytest test runner

## Need Help?

- **Documentation**: Check the main README.md
- **Issues**: Search existing issues or create new ones
- **Discussions**: Use GitHub discussions for questions
- **Code**: Review existing code for patterns and examples

Thank you for contributing! 🚀
