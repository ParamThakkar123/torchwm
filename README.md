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

## Supported Algorithms

| Algorithm | Description | Key Features |
|-----------|-------------|--------------|
| **Dreamer** | Model-based RL with latent dynamics | Imagination, actor-critic |
| **JEPA** | Self-supervised visual representations | Masked prediction, ViT |
| **IRIS** | Sample-efficient RL with Transformers | Discrete VAEs, world models |
| **DiT** | Diffusion Transformer workflows | Patch embeddings, diffusion backbones |

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
