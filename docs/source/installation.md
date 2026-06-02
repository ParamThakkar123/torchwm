# Installation

TorchWM supports multiple installation methods depending on your use case.

## From PyPI

For stable releases:

```bash
# CPU build (default core dependencies: torch, torchvision, torchaudio, gym, etc.)
pip install torchwm

# GPU build (CUDA-enabled PyTorch stack)
pip install torchwm[gpu]

# With specific extras
pip install torchwm[gym]       # Additional gym environments (huggingface-hub, pygame, autorom)
pip install torchwm[ml-agents] # Unity ML-Agents support
pip install torchwm[ml]        # TensorBoard, Weights & Biases, logging tools
pip install torchwm[viz]       # FastAPI, Uvicorn, documentation tools
pip install torchwm[docs]      # Sphinx and documentation tools
pip install torchwm[dev]       # Testing and development tools (pytest, mypy, pre-commit)

# Install multiple extras
pip install torchwm[gpu,gym,ml-agents,dev]
```

### Available Extras

| Extra | Description |
|-------|-------------|
| `gpu` | CUDA-enabled PyTorch, torchvision, and torchaudio stack |
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

## CPU and CUDA PyTorch Selection

TorchWM installs the CPU-compatible PyTorch stack by default so `pip install torchwm`
works consistently on machines without a GPU. If a GPU is available and you want
CUDA acceleration, request the GPU extra:

```bash
pip install torchwm[gpu]
```

When installing with `uv`, TorchWM's project metadata no longer forces the CUDA
PyTorch index for default installs, and routes the `gpu` extra to the CUDA 12.1
PyTorch wheel index. If your environment needs a different CUDA wheel family,
install the matching PyTorch packages from the official PyTorch index before
installing TorchWM.

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