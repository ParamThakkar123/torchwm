# TorchWM

<div align="center">
  <img src="docs/images/torchwm-logo.svg" alt="TorchWM Logo" height="80">
  <p>
    <a href="https://pypi.org/project/torchwm/"><img alt="PyPI version" src="https://badge.fury.io/py/torchwm.svg"></a>
    <a href="https://pypi.org/project/torchwm/"><img alt="PyPI downloads" src="https://img.shields.io/pypi/dm/torchwm.svg"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
    <a href="https://paramthakkar123.github.io/torchwm/"><img alt="Documentation" src="https://img.shields.io/badge/docs-link-blue.svg"></a>
  </p>
  <p><strong>Modular PyTorch Library for World Models</strong></p>
</div>

---

## ⚡ Quick Start

Train Dreamer agents in just 3 lines of code. TorchWM provides a unified interface for training and deploying world models.

### Installation

```bash
# Core dependencies
pip install torchwm

# With extras
pip install torchwm[gym]       # Additional gym environments
pip install torchwm[ml-agents] # Unity ML-Agents
pip install torchwm[ml]        # TensorBoard, W&B logging
pip install torchwm[viz]       # FastAPI visualization
pip install torchwm[dev]       # Testing and linting
```

### Training a Dreamer Agent

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

The lower-level research modules are still available when you need direct
control:

```python
from torchwm import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env = "walker-walk"
agent = DreamerAgent(cfg)
```

### Creating Environments and Operators

```python
import torchwm

env = torchwm.make_env("CartPole-v1", backend="gym")
op = torchwm.get_operator("dreamer", image_size=64, action_dim=6)
processed = op.process({"image": image, "action": action})
```

## 🚀 Features

- 🎯 **Unified Interface**: Consistent API across all world model algorithms
- 🔧 **Modular Components**: Swappable encoders, decoders, and backbones
- 🚀 **High Performance**: Optimized for both training and inference
- 🌍 **Multi-Environment**: Support for DMC, Gym, Unity, and custom environments
- 📊 **Rich Monitoring**: Integrated logging with Weights & Biases and TensorBoard
- 🧠 **Research Ready**: Easy experimentation with different architectures

## 🧠 Supported Algorithms

| Algorithm | Description | Key Features |
|-----------|-------------|--------------|
| **Dreamer** | Model-based RL with latent dynamics | Imagination, actor-critic |
| **JEPA** | Self-supervised visual representations | Masked prediction, ViT |
| **IRIS** | Sample-efficient RL with Transformers | Discrete VAEs, world models |
| **Diamond** | Diffusion + RL for continuous control | EDM sampling, value functions |

## 📖 Documentation

📖 [Full Documentation](https://paramthakkar123.github.io/torchwm/)

### Get Started
- [Installation](https://paramthakkar123.github.io/torchwm/getting_started.html)
- [Training Guide](https://paramthakkar123.github.io/torchwm/training_guide.html)
- [Inference Guide](https://paramthakkar123.github.io/torchwm/inference_guide.html)

### User Guides
- [Operators Guide](https://paramthakkar123.github.io/torchwm/operators_guide.html)
- [Environments Guide](https://paramthakkar123.github.io/torchwm/environments_guide.html)
- [Package Overview](https://paramthakkar123.github.io/torchwm/package_overview.html)

### Algorithms
- [Dreamer](https://paramthakkar123.github.io/torchwm/dreamer.html)
- [JEPA](https://paramthakkar123.github.io/torchwm/jepa.html)
- [IRIS](https://paramthakkar123.github.io/torchwm/iris.html)
- [DiT](https://paramthakkar123.github.io/torchwm/dit.html)

## 🤝 Community

- 🐛 [Issue Tracker](https://github.com/paramthakkar123/torchwm/issues)
- 💬 [Discussions](https://github.com/paramthakkar123/torchwm/discussions)
- 📧 [PyPI](https://pypi.org/project/torchwm/)

---

```{important}
TorchWM is under active development. APIs may change between versions.
```

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

## ⚡ Quick Start

TorchWM provides a unified interface for training and deploying world models.

### Installation

```bash
pip install torchwm
# or with uv
uv add torch torchvision torchaudio
```

### Training a Dreamer Agent

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

The lower-level research modules are still available when you need direct
control:

```python
from torchwm import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env = "walker-walk"
agent = DreamerAgent(cfg)
```

### Creating Environments and Operators

```python
import torchwm

env = torchwm.make_env("CartPole-v1", backend="gym")
op = torchwm.get_operator("dreamer", image_size=64, action_dim=6)
processed = op.process({"image": image, "action": action})
```

## Features

- 🎯 **Unified Interface**: Consistent API across all world model algorithms
- 🔧 **Modular Components**: Swappable encoders, decoders, and backbones
- 🚀 **High Performance**: Optimized for both training and inference
- 🌍 **Multi-Environment**: Support for DMC, Gym, Unity, and custom environments
- 📊 **Rich Monitoring**: Integrated logging with Weights & Biases and TensorBoard
- 🧠 **Research Ready**: Easy experimentation with different architectures

## 🧠 Supported Algorithms

| Algorithm | Description | Key Features |
|-----------|-------------|--------------|
| **Dreamer** | Model-based RL with latent dynamics | Imagination, actor-critic |
| **JEPA** | Self-supervised visual representations | Masked prediction, ViT |
| **IRIS** | Sample-efficient RL with Transformers | Discrete VAEs, world models |
| **Diamond** | Diffusion + RL for continuous control | EDM sampling, value functions |

## 🤝 Community

- 📖 [Documentation](https://paramthakkar123.github.io/torchwm/)
- 🐛 [Issue Tracker](https://github.com/paramthakkar123/torchwm/issues)
- 💬 [Discussions](https://github.com/paramthakkar123/torchwm/discussions)
- 📧 [PyPI](https://pypi.org/project/torchwm/)

---

```{important}
TorchWM is under active development. APIs may change between versions.
```