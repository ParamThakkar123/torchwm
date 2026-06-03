# Contributing

We welcome contributions to TorchWM! This guide covers how to get started.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ParamThakkar123/torchwm.git
   cd torchwm
   ```

2. **Install in development mode**
   ```bash
   uv sync --dev
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pre-commit** for automated checks

Run checks:
```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy .

# All checks
pre-commit run --all-files
```

## Testing

Run the test suite:
```bash
# All tests
pytest

# Specific tests
pytest tests/test_operators.py

# With coverage
pytest --cov=world_models --cov-report=html
```

## Documentation

Build docs locally:
```bash
cd docs
sphinx-build -b html source build/html
```

Open `docs/build/html/index.html` in your browser.

## Adding New Features

### 1. New Operators

```python
from world_models.inference.operators.base import OperatorABC

class NewOperator(OperatorABC):
    def process(self, inputs):
        # Your preprocessing logic
        return processed_tensors
```

Add to `__init__.py` and create tests.

### 2. New Models

1. Create model class in `world_models/models/`
2. Add config class in `world_models/configs/`
3. Add operator in `world_models/inference/operators/`
4. Update training scripts
5. Add documentation and tests

### 3. New Environments

1. Implement environment wrapper in `world_models/envs/`
2. Add to environment registry
3. Update documentation

## Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make your changes**
4. **Run tests and checks**
   ```bash
   pre-commit run --all-files
   pytest
   ```
5. **Update documentation** if needed
6. **Commit your changes**
   ```bash
   git commit -m "Add feature: my feature"
   ```
7. **Push to your fork**
   ```bash
   git push origin feature/my-feature
   ```
8. **Create a Pull Request**

## Commit Messages

Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation
- `test:` Testing
- `refactor:` Code refactoring
- `chore:` Maintenance

Examples:
- `feat: add VLA operator support`
- `fix: resolve memory leak in dreamer training`
- `docs: update operators guide`

## Issue Reporting

- **Bug reports**: Use the bug report template
- **Feature requests**: Use the feature request template
- **Questions**: Use discussions

## Code of Conduct

Please follow our code of conduct in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.