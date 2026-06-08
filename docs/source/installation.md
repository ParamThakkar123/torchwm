# Installation

TorchWM supports multiple installation methods depending on your use case.

## From PyPI

For stable releases:

```bash
# Minimal library install. This keeps managed notebook runtimes from pulling in
# every simulator, visualization, and benchmark dependency at once.
pip install torchwm

# Add the backends you need for your tutorial or experiment.
pip install "torchwm[dmc]"        # DeepMind Control Suite / Dreamer DMC tutorials
pip install "torchwm[gym]"        # Gym/Gymnasium image environments
pip install "torchwm[atari]"      # ALE Atari environments
pip install "torchwm[mujoco]"     # MuJoCo / Gymnasium MuJoCo tasks
pip install "torchwm[robotics]"   # Gymnasium Robotics tasks
pip install "torchwm[brax]"       # Brax tasks
pip install "torchwm[datasets]"   # HDF5 / Hugging Face dataset helpers
pip install "torchwm[ml-agents]"  # Unity ML-Agents support
pip install "torchwm[ml]"         # TensorBoard, Weights & Biases, logging tools
pip install "torchwm[viz]"        # FastAPI/Uvicorn visualization server
pip install "torchwm[docs]"       # Sphinx and documentation tools
pip install "torchwm[dev]"        # Testing and development tools

# Install multiple extras.
pip install "torchwm[dmc,atari,ml]"
```

### Available Extras

| Extra | Description |
|-------|-------------|
| `dmc` | DeepMind Control Suite dependencies for DMC Dreamer tutorials |
| `gym` | Gym/Gymnasium image environments and video helpers |
| `atari` | ALE Atari environments and preprocessing helpers |
| `mujoco` | MuJoCo / Gymnasium MuJoCo task support |
| `robotics` | Gymnasium Robotics support |
| `brax` | Brax support |
| `datasets` | HDF5 and Hugging Face dataset helpers |
| `ml-agents` | Unity ML-Agents support |
| `ml` | TensorBoard, Weights & Biases, and analysis tools |
| `viz` | FastAPI and Uvicorn for the visualization server |
| `docs` | Sphinx and documentation building tools |
| `dev` | pytest, mypy, pre-commit, and browser-test tooling |

### Core Dependencies

The minimal installation includes only the common Python and PyTorch runtime pieces: `torch`, `torchvision`, `einops`, `pyyaml`, `tqdm`, `requests`, and `click`. Environment backends, dataset downloads, browser testing, and visualization dependencies live in extras so `pip install torchwm` does not replace large portions of preinstalled notebook runtimes.

## Managed notebooks such as Kaggle or Colab

Managed notebook images often come with pinned CUDA, PyTorch, NumPy, and simulator packages. If a runtime already contains most scientific dependencies, prefer either a fresh virtual environment or a minimal install that preserves the platform stack:

```bash
# Preserve the managed runtime's pinned packages.
pip install --no-deps torchwm

# Then add only the backend needed by the notebook.
pip install "einops>=0.8.2" "pyyaml>=6.0.3" "tqdm>=4.67.1" "requests>=2.32.0" "click>=8.0.0"
pip install "dm-control>=1.0.28" "mujoco>=3.3.1" "gymnasium>=1.2.2" "opencv-python>=4.12.0.88" "moviepy>=2.2.1"
```

`torchwm[dmc]` declares both `dm-control` and `mujoco`; if `pip show dm-control mujoco` reports either package missing after installation, upgrade to a TorchWM release that includes the `dmc` extra or install from this repository. After changing MuJoCo or `dm-control` versions in a running notebook, restart the kernel before importing `torchwm` or `dm_control`. Attribute errors inside `dm_control.mujoco` such as missing fields on `MjData` usually indicate a `dm-control`/`mujoco` ABI mismatch from mixed package versions.

## From Source

For the latest development version:

```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm

# Core dependencies
pip install -e .

# With extras
pip install -e ".[dmc,atari,datasets,ml-agents,ml,viz,dev,docs]"
```

## CUDA Support

For GPU acceleration, install PyTorch with CUDA before installing TorchWM when your platform does not already provide it:

```bash
# Using uv (recommended)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or using pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
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
