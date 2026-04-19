# Installation

TorchWM supports multiple installation methods depending on your use case.

## From PyPI

For stable releases:

```bash
pip install torchwm
```

## From Source

For the latest development version:

```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm
pip install -e .
```

## Development Installation

For development, testing, and documentation:

```bash
pip install -e ".[dev]"
```

This installs additional dependencies for:
- Testing (`pytest`, `pytest-cov`)
- Documentation (`sphinx`, `myst-parser`)
- Development tools (`pre-commit`, `ruff`, `mypy`)

## CUDA Support

For GPU acceleration, install PyTorch with CUDA:

```bash
# Using uv (recommended)
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or using pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Requirements

### Core Dependencies
- Python >= 3.8
- PyTorch >= 2.0
- NumPy
- Pillow

### Optional Dependencies
- `gymnasium`: For Gym environments
- `dm-control`: For DeepMind Control Suite
- `wandb`: For experiment logging
- `opencv-python`: For video processing
- `selenium`: For UI testing

## Docker

Build and run using Docker:

```bash
# Build
docker build -t torchwm .

# Run
docker run -it torchwm
```

## Verification

Verify your installation:

```python
import torch
import world_models

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("TorchWM imported successfully!")
```