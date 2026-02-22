# TorchWM

TorchWM is a modular PyTorch library for world models and latent dynamics learning.
It includes practical implementations of Dreamer-style agents, PlaNet/RSSM utilities,
JEPA-style representation learning, and diffusion/transformer building blocks.

## Highlights

- Modular components for encoders, decoders, RSSMs, reward/value heads, and policies
- Multiple environment backends: DMC, Gym/Gymnasium, Atari, MuJoCo, Unity ML-Agents
- Replay/memory utilities for both Dreamer and PlaNet-style training loops
- ViT + masking utilities for JEPA workflows
- Diffusion utilities (DDPM schedule + DiT model)

## Installation

Install from PyPI:

```bash
pip install torchwm
```

Install from source:

```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm
pip install -e .
```

Development extras:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Train Dreamer on Gym

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"  # dmc | gym | unity_mlagents
cfg.env = "Pendulum-v1"
cfg.total_steps = 10_000

agent = DreamerAgent(cfg)
agent.train()
```

### Train Dreamer on Unity ML-Agents

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "unity_mlagents"
cfg.unity_file_name = r"E:\UnityBuilds\MyEnv.exe"
cfg.unity_behavior_name = "MyBehavior"
cfg.unity_no_graphics = True
cfg.unity_time_scale = 20.0

agent = DreamerAgent(cfg)
agent.train()
```

### Train JEPA

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

## Documentation

- Sphinx source: `docs/source`
- Getting started guide: `docs/source/getting_started.md`
- Package overview: `docs/source/package_overview.md`
- API reference (autodoc): `docs/source/api_reference.rst`

Build HTML docs locally:

```bash
sphinx-build -b html docs/source docs/build/html
```

## Package Layout

- `world_models/models`: Agents and model architectures (`Dreamer`, `JEPAAgent`, `Planet`, ViT, diffusion)
- `world_models/configs`: Config classes (`DreamerConfig`, `JEPAConfig`, `DiTConfig`)
- `world_models/envs`: Environment adapters and wrappers
- `world_models/training`: Script-style training entrypoints
- `world_models/datasets`: CIFAR10/ImageNet/ImageFolder data loaders
- `world_models/memory`: Replay and episodic memory implementations
- `world_models/utils`: Training/logging/distributed helper utilities

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{Thakkar_GitHub_-_ParamThakkar123_torchwm,
author = {Thakkar, Param},
title = {{GitHub - ParamThakkar123/torchwm: A modular PyTorch library designed for learning, training, and deploying world models across various environments.}},
year = {2025},
url = {https://github.com/ParamThakkar123/torchwm}
}
```

Package: [torchwm on PyPI](https://pypi.org/project/torchwm/)
