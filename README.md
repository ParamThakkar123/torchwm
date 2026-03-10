# TorchWM

TorchWM is a modular PyTorch library for world models and latent dynamics learning.
It includes practical implementations of Dreamer-style agents, PlaNet/RSSM utilities,
JEPA-style representation learning, and diffusion/transformer building blocks.

## Highlights

- Modular components for encoders, decoders, RSSMs, reward/value heads, and policies
- Multiple environment backends: DMC, Gym/Gymnasium, Atari, MuJoCo, Unity ML-Agents, ProcGen
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

## Package Layout

```
torchwm/
├── world_models/              # Main package
│   ├── models/               # Agent implementations (Dreamer, JEPA, etc.)
│   ├── configs/              # Configuration classes
│   ├── envs/                 # Environment adapters (DMC, Gym, ProcGen, Unity)
│   ├── utils/                # Utilities and helpers
│   └── __init__.py
├── tests/                     # Test suite
│   └── test_env_adapters.py  # Environment integration tests
├── docs/                      # Sphinx documentation
│   └── source/
├── .github/                   # CI/CD workflows
├── pyproject.toml             # Project configuration
├── CONTRIBUTING.md            # Contribution guidelines
└── README.md                  # This file
```

## Quick Start

### Train Dreamer on Gym

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"  # dmc | gym | procgen | unity_mlagents
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

### Train Dreamer on ProcGen

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "procgen"
cfg.env = "coinrun"
cfg.total_steps = 10_000

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

## Testing

This project uses separate CI workflows for different environment backends to ensure compatibility:

- **DMC Tests**: Tests DeepMind Control Suite integration
- **Gym Tests**: Tests Gym/Gymnasium environment support
- **ProcGen Tests**: Tests OpenAI ProcGen environment support (Python 3.10)
- **Unity ML-Agents Tests**: Tests Unity ML-Agents integration
- **General Environment Tests**: Tests environment wrappers and utilities

Run environment-specific tests locally:
```bash
# DMC environments
pytest tests/test_env_adapters.py::test_make_env_dmc_backend -v

# Gym environments
pytest tests/test_env_adapters.py::test_make_env_gym_backend tests/test_env_adapters.py::test_gym_image_env_discrete_action_mapping -v

# ProcGen environments
pytest tests/test_env_adapters.py::test_make_procgen_env tests/test_env_adapters.py::test_list_available_procgen_envs tests/test_env_adapters.py::test_make_env_procgen_backend -v

# Unity ML-Agents environments
pytest tests/test_env_adapters.py::test_make_env_unity_backend -v
```

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
