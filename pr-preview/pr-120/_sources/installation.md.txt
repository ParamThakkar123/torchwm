# Installation

TorchWM supports multiple installation methods depending on your use case.

## From PyPI

For stable releases:

```bash
# Minimal installation (core dependencies only)
pip install torchwm

# With specific extras
pip install torchwm[gym]      # Gym environments support
pip install torchwm[viz]      # Visualization tools
pip install torchwm[ml]       # Machine learning utilities
pip install torchwm[docs]     # Documentation tools
pip install torchwm[dev]      # Development tools (testing, linting)

# Install multiple extras
pip install torchwm[gym,viz]
```

### Available Extras

| Extra | Description |
|-------|-------------|
| `gym` | Gymnasium and DM Control environments |
| `viz` | Visualization and plotting tools |
| `ml` | Additional ML utilities and helpers |
| `docs` | Documentation building tools |
| `dev` | Development tools (pytest, ruff, mypy, pre-commit) |

The minimal installation includes core dependencies: torch, torchvision, torchaudio, einops, pyyaml, tqdm.

## From Source

For the latest development version:

```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm

# Minimal installation
pip install -e .

# With extras
pip install -e ".[gym,viz,dev]"
```

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
- torch, torchvision, torchaudio
- einops, pyyaml, tqdm
- NumPy, Pillow

Optional dependencies are installed via extras (see Available Extras above).

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

```python :class: thebe
import torch
import world_models

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("TorchWM imported successfully!")
```