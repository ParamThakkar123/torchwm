# TorchWM

<div align="center">
  <p>
    <a href="https://pypi.org/project/torchwm/"><img alt="PyPI version" src="https://badge.fury.io/py/torchwm.svg"></a>
    <a href="https://pypi.org/project/torchwm/"><img alt="PyPI downloads" src="https://img.shields.io/pypi/dm/torchwm.svg"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
    <a href="https://paramthakkar123.github.io/torchwm/"><img alt="Documentation" src="https://img.shields.io/badge/docs-link-blue.svg"></a>
  </p>
  <p><strong>Modular PyTorch library for world models, latent dynamics, and representation learning.</strong></p>
</div>

TorchWM provides reusable PyTorch components and training utilities for Dreamer-style agents, JEPA representations, IRIS, DiT, and related world-model workflows.

## Quick Start

```bash
# Install the core package from PyPI.
pip install torchwm

# With extras
pip install torchwm[gym]       # Additional gym environments
pip install torchwm[procgen]   # Procgen benchmark environments
pip install torchwm[ml-agents] # Unity ML-Agents
pip install torchwm[ml]        # TensorBoard, W&B logging
pip install torchwm[viz]       # FastAPI visualization
pip install torchwm[dev]       # Testing and linting

# Or add it to a uv-managed project.
uv add torchwm
```

TorchWM depends on PyTorch but does not force a single PyTorch wheel index. If you need a specific PyTorch build, install or add the PyTorch packages with the index recommended for your platform by the [PyTorch installation selector](https://pytorch.org/get-started/locally/):

```bash
# Example: CUDA 12.1 wheels. Choose a different index for CPU, ROCm, CUDA 11.x, CUDA 12.4+, or macOS.
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu121
```

Use the friendly top-level API for the common path:

```python
import torchwm

agent = torchwm.create_model(
    "dreamer",
    env="walker-walk",
    total_steps=1_000_000,
)
agent.train()
```

## Features

- Unified interfaces across world-model algorithms
- Modular encoders, decoders, dynamics models, and backbones
- Training and inference utilities for model-based reinforcement learning
- Environment integrations for Gym/Gymnasium, Unity ML-Agents, MuJoCo, Brax, and robotics extras
- Optional logging, visualization, development, and documentation extras


### Visualization Trackers

Project RSSM latent trajectories and JEPA/ViT embeddings to 2D or 3D for quick
inspection in notebooks or saved Plotly HTML reports:

```python
import numpy as np
import torchwm

states = np.random.default_rng(0).normal(size=(2, 6, 32))
projection = torchwm.project_latent_trajectories(states, method="pca")
torchwm.plot_projection(projection, output_path="rssm_latents.html")

embeddings = np.random.default_rng(1).normal(size=(128, 768))
labels = np.array(["cat"] * 64 + ["dog"] * 64)
projection = torchwm.project_representation_embeddings(embeddings, labels=labels)
torchwm.plot_projection(projection, output_path="jepa_embeddings.html")
```

See `docs/source/visualization.md` and
`examples/visualization_trackers_example.py` for complete examples.

## Supported Algorithms

| Algorithm | Description | Key Features |
|-----------|-------------|--------------|
| **Dreamer** | Model-based RL with latent dynamics | Imagination, actor-critic |
| **JEPA** | Self-supervised visual representations | Masked prediction, ViT |
| **IRIS** | Sample-efficient RL with Transformers | Discrete VAEs, world models |
| **DiT** | Diffusion Transformer workflows | Patch embeddings, diffusion backbones |
| **DIAMOND** | Diffusion world model for pixel-control RL | EDM sampling, Atari imagination rollouts |

## Documentation

- [Full Documentation](https://paramthakkar123.github.io/torchwm/)
- [Installation Guide](https://paramthakkar123.github.io/torchwm/installation.html)
- [Training Guide](https://paramthakkar123.github.io/torchwm/training_guide.html)
- [API Reference](https://paramthakkar123.github.io/torchwm/api_reference.html)

## Community

- [Issue Tracker](https://github.com/paramthakkar123/torchwm/issues)
- [Discussions](https://github.com/paramthakkar123/torchwm/discussions)
- [PyPI](https://pypi.org/project/torchwm/)

> TorchWM is under active development. APIs may change between versions.
