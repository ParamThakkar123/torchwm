# Contributing to TorchWM

We welcome contributions to TorchWM! This document outlines the guidelines for contributing to the project.

## Ways to Contribute

- **Report Bugs**: Open an issue on GitHub with a clear description of the bug, steps to reproduce, and expected behavior.
- **Suggest Features**: Propose new features or improvements via GitHub issues.
- **Submit Pull Requests**: Implement fixes or enhancements and submit a PR.

## Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/torchwm.git
   cd torchwm
   ```

2. Install dependencies with development extras:
   ```bash
   uv sync --dev
   ```

3. Set up pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

If you are not using `uv`, install the project in editable mode with development extras instead:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

## Coding Standards

- Follow PEP 8 style guidelines.
- Use type hints where possible.
- Format code with Black: `uv run black .`
- Lint code with Ruff: `uv run ruff check .`
- Type-check code with MyPy where practical: `uv run mypy .`
- Run all configured hooks with: `uv run pre-commit run --all-files`

## Testing

- Run tests with: `uv run pytest`
- Run a fast subset with: `uv run pytest -m "not slow and not gpu and not integration"`
- The repository pins Python bytecode caches outside the source tree with `PYTHONPYCACHEPREFIX=build/__pycache__`; keep this set when running ad-hoc Python commands so `__pycache__` directories are not created beside source files.
- Ensure all tests pass before submitting a PR.
- Add tests for new features or bug fixes.

## Pull Request Process

1. Create a feature branch from `main`: `git checkout -b feature/your-feature`
2. Make your changes and commit them.
3. Run checks and tests: `uv run pre-commit run --all-files` and `uv run pytest`
4. Push to your fork and create a PR.
5. PRs must pass CI checks and have at least one approval.

## Code of Conduct

Please be respectful and inclusive. Harassment or discriminatory behavior will not be tolerated.

For questions, reach out via GitHub issues.
