# TorchWM

<div align="center">
  <img src="docs/images/torchwm-logo.svg" alt="TorchWM Logo" height="80">
  <p><strong>Modular PyTorch Library for World Models</strong></p>
</div>

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
# Returns standardized tensors for inference
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

## Documentation

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

## Community

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

## Quick Start

TorchWM provides a unified interface for training and deploying world models.

### Installation

```bash
pip install torchwm
# or with uv
uv add torch torchvision torchaudio
```

## Training Scripts

TorchWM includes several training scripts for different world model components. These can be run directly from the command line.

### Train World Model Pipeline

Train a complete world model (VAE + MDNRNN + Controller) on any Gym environment:

```bash
# Train on CarRacing
python -m world_models.training.train_world_model --env CarRacing-v2

# Train on Pendulum
python -m world_models.training.train_world_model --env Pendulum-v1

# Custom data/log directories
python -m world_models.training.train_world_model --env YourEnv-v0 --data_dir ./my_data --logdir ./my_logs

# Specify action size manually if env loading fails
python -m world_models.training.train_world_model --env BipedalWalker-v3 --action_size 4

# Test trained model
python -m world_models.training.train_world_model --env CarRacing-v2 --test
```

### Other Training Scripts

- **ConvVAE**: `python -m world_models.training.train_convvae`
- **MDNRNN**: `python -m world_models.training.train_mdn_rnn`
- **Controller**: `python -m world_models.training.train_controller`
- **Planet**: `python -m world_models.training.train_planet`
- **RSSM**: `python -m world_models.training.train_rssm`
- **JEPA**: `python -m world_models.training.train_jepa`

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