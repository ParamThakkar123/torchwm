# Installation

TorchWM supports multiple installation methods depending on your use case.

## From PyPI

For stable releases:

```bash
# Core dependencies (torch, torchvision, torchaudio, gym, gymnasium, etc.)
pip install torchwm

# With specific extras
pip install torchwm[gym]       # Additional gym environments (huggingface-hub, pygame, autorom)
pip install torchwm[ml-agents] # Unity ML-Agents support
pip install torchwm[ml]        # TensorBoard, Weights & Biases, logging tools
pip install torchwm[viz]       # FastAPI, Uvicorn, documentation tools
pip install torchwm[docs]      # Sphinx and documentation tools
pip install torchwm[dev]       # Testing and development tools (pytest, mypy, pre-commit)

# Install multiple extras
pip install torchwm[gym,ml-agents,dev]
```

### Available Extras

| Extra | Description |
|-------|-------------|
| `gym` | Additional Gym environment dependencies (huggingface-hub, pygame, autorom) |
| `ml-agents` | Unity ML-Agents support |
| `ml` | TensorBoard, Weights & Biases, and logging tools |
| `viz` | FastAPI, Uvicorn for visualization server |
| `docs` | Sphinx and documentation building tools |
| `dev` | pytest, mypy, pre-commit for development |

### Core Dependencies

The minimal installation includes: torch, torchvision, torchaudio, einops, pyyaml, tqdm, opencv-python, requests, gym, gymnasium, moviepy, h5py, plotly, ale-py, selenium, scikit-learn, umap-learn.

## From Source

For the latest development version:

```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm

# Core dependencies
pip install -e .

# With extras
pip install -e ".[gym,ml-agents,ml,viz,dev,docs]"
```

## CUDA Support

For GPU acceleration, install PyTorch with CUDA:

```bash
# Using uv (recommended)
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or using pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

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
import torchwm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("TorchWM imported successfully!")
```