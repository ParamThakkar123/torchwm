# Getting Started

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

For development and tests:

```bash
pip install -e ".[dev]"
```

## Quick Start: Dreamer

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"
cfg.total_steps = 10_000

agent = DreamerAgent(cfg)
agent.train()
```

## Quick Start: JEPA

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

## Environment Backends

Dreamer supports multiple backends through `DreamerConfig.env_backend`:

- `dmc`: DeepMind Control Suite tasks (for example `walker-walk`)
- `gym`: Gym/Gymnasium environment IDs or an existing environment instance
- `unity_mlagents`: Unity ML-Agents executable environments

Important Unity settings are available in `DreamerConfig`:
- `unity_file_name`
- `unity_behavior_name`
- `unity_no_graphics`
- `unity_time_scale`

## Typical Training Flow

1. Create a config object (`DreamerConfig` or `JEPAConfig`).
2. Override dataset/environment and optimization fields.
3. Instantiate the corresponding agent (`DreamerAgent`, `JEPAAgent`).
4. Call `train()` and monitor logs/checkpoints.

For complete API details, see {doc}`api_reference`.
