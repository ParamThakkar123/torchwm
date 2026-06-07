PYTHON ?= python
export PYTHONPYCACHEPREFIX ?= build/__pycache__

.PHONY: test lint format

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
