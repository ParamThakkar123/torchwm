# TorchWM 🧠

[![PyPI version](https://badge.fury.io/py/torchwm.svg)](https://pypi.org/project/torchwm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

TorchWM is a modular PyTorch library for world models and latent dynamics learning. It provides practical implementations of Dreamer-style agents, PlaNet/RSSM utilities, JEPA-style representation learning, and diffusion/transformer building blocks for reinforcement learning and generative modeling.

## ✨ Highlights

- 🧩 **Modular Components**: Encoders, decoders, RSSMs, reward/value heads, and policies
- 🔄 **Modular RSSM**: Swappable encoder/decoder/backbone for research experiments
- 🌍 **Multiple Environment Backends**: DMC, Gym/Gymnasium, Atari, MuJoCo, Unity ML-Agents, **Isaac Lab**, **Brax**
- 🚀 **GPU-Accelerated Vectorized Environments**: CUDA integration and async batching for Isaac Lab and Brax
- 📦 **Memory & Replay Utilities**: For Dreamer and PlaNet-style training loops
- 🖼️ **ViT + Masking Utilities**: For JEPA workflows
- 🌊 **Diffusion Utilities**: DDPM schedule + DiT model
- 🏃 **Ready-to-Use Agents**: Dreamer, PlaNet, IRIS, JEPA implementations

## 📦 Installation

### Install from PyPI
```bash
pip install torchwm
```

### Install from Source
```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm
pip install -e .
```

### Development Extras
```bash
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Train Dreamer on Gym
```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"  # dmc | gym | unity_mlagents | isaaclab
cfg.env = "Pendulum-v1"
cfg.total_steps = 10_000

agent = DreamerAgent(cfg)
agent.train()
```

### Train Dreamer on Isaac Lab (GPU Vectorized) ⚡
```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig
from world_models.envs import GPUVectorizedEnv
import isaaclab.envs

# Create GPU vectorized env factory
env_factory = lambda num_envs=None, device=None, seed=None: isaaclab.envs.ManagerBasedRLEnv({
    "task": "IsaacCartpole-v0",
    "num_envs": num_envs,
    "device": device.type,
    "seed": seed,
})

cfg = DreamerConfig()
cfg.env_backend = "isaaclab"
cfg.env = GPUVectorizedEnv(env_factory, num_envs=32, device="cuda")  # Prebuilt env
cfg.total_steps = 10_000

agent = DreamerAgent(cfg)
agent.train()
```

### Train Dreamer on Brax (GPU Vectorized) ⚡
```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig
from world_models.envs import GPUVectorizedEnv
import brax.envs

# Create GPU vectorized env factory
env_factory = lambda num_envs=None, device=None, seed=None: brax.envs.create('ant', batch_size=num_envs)

cfg = DreamerConfig()
cfg.env_backend = "brax"
cfg.env = GPUVectorizedEnv(env_factory, num_envs=32, device="cuda")
cfg.total_steps = 10_000

agent = DreamerAgent(cfg)
agent.train()
```

### Train JEPA on Images
```python
from world_models.models import JEPAAgent
from world_models.configs import JEPAConfig

cfg = JEPAConfig()
cfg.dataset = "imagefolder"
cfg.root_path = "./data"
cfg.image_folder = "train"
cfg.epochs = 10

agent = JEPAAgent(cfg)
agent.train()
```

## 📚 Documentation

- **Getting Started Guide**: [`docs/source/getting_started.md`](docs/source/getting_started.md)
- **Package Overview**: [`docs/source/package_overview.md`](docs/source/package_overview.md)
- **API Reference**: [`docs/source/api_reference.rst`](docs/source/api_reference.rst)

### Build Docs Locally
```bash
sphinx-build -b html docs/source docs/build/html
```

## 🏗️ Package Layout

- `world_models/models/`: Agents and architectures (Dreamer, JEPAAgent, Planet, IRIS, ViT, diffusion)
- `world_models/models/modular_rssm/`: Modular RSSM with swappable components
- `world_models/configs/`: Config classes (DreamerConfig, JEPAConfig, IRISConfig, DiTConfig)
- `world_models/envs/`: Environment adapters, wrappers, **GPU-accelerated vectorized environments**
- `world_models/training/`: Script-style training entrypoints
- `world_models/datasets/`: CIFAR10/ImageNet/ImageFolder data loaders
- `world_models/memory/`: Replay and episodic memory implementations
- `world_models/utils/`: Training/logging/distributed helper utilities

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

## 📖 Citation

```bibtex
@misc{Thakkar_GitHub_-_ParamThakkar123_torchwm,
  author = {Thakkar, Param},
  title = {{TorchWM: A modular PyTorch library for world models and latent dynamics learning}},
  year = {2025},
  url = {https://github.com/ParamThakkar123/torchwm}
}
```

🔗 **Package**: [torchwm on PyPI](https://pypi.org/project/torchwm/)  
🔗 **Repository**: [GitHub](https://github.com/ParamThakkar123/torchwm)
