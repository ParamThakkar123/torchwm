# TorchWM

<div style="text-align: center; margin: 2rem 0;">
    <img src="_static/torchwm-logo.svg" alt="TorchWM Logo" style="height: 80px; width: auto;">
    <p style="font-size: 1.2rem; color: var(--pst-color-text-muted); margin-top: 1rem;">
        Modular PyTorch Library for World Models
    </p>
</div>

```{toctree}
:hidden:
:maxdepth: 1
:caption: Get Started

getting_started
installation
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: User Guides

operators_guide
training_guide
inference_guide
environments_guide
package_overview
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Algorithms

dreamer
jepa
iris
dit
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Reference

api_reference
configs_reference
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Development

contributing
benchmarks
```

---

## Quick Start

TorchWM provides a unified interface for training and deploying world models.

### Installation

```bash
pip install torchwm
# or with uv
uv add torch torchvision torchaudio
```

### Training a Dreamer Agent

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env = "walker-walk"
cfg.total_steps = 1_000_000

agent = DreamerAgent(cfg)
agent.train()
```

### Using Inference Operators

```python
from world_models.inference.operators import DreamerOperator

op = DreamerOperator(image_size=64, action_dim=6)
processed = op.process({'image': image, 'action': action})
```

## Features

- 🎯 **Unified Interface**: Consistent API across all world model algorithms
- 🔧 **Modular Components**: Swappable encoders, decoders, and backbones
- 🚀 **High Performance**: Optimized for both training and inference
- 🌍 **Multi-Environment**: Support for DMC, Gym, Unity, and custom environments
- 📊 **Rich Monitoring**: Integrated logging with Weights & Biases and TensorBoard
- 🧠 **Research Ready**: Easy experimentation with different architectures

## Supported Algorithms

| Algorithm | Description | Key Features |
|-----------|-------------|--------------|
| **Dreamer** | Model-based RL with latent dynamics | Imagination, actor-critic |
| **JEPA** | Self-supervised visual representations | Masked prediction, ViT |
| **IRIS** | Sample-efficient RL with Transformers | Discrete VAEs, world models |
| **Diamond** | Diffusion + RL for continuous control | EDM sampling, value functions |

## Community

- 📖 [Documentation](https://paramthakkar123.github.io/torchwm/)
- 🐛 [Issue Tracker](https://github.com/paramthakkar123/torchwm/issues)
- 💬 [Discussions](https://github.com/paramthakkar123/torchwm/discussions)
- 📧 [PyPI](https://pypi.org/project/torchwm/)

---

```{important}
TorchWM is under active development. APIs may change between versions.
```