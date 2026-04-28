# Package Overview

TorchWM is organized into focused modules so you can use only the pieces you need.

## Quick Import (Public API)

Import everything you need from a single namespace:

```python
import world_models
# or
from world_models import DreamerAgent, DreamerConfig
```

### Available Exports

| Category | Exports |
|----------|--------|
| **Models** | `Dreamer`, `Planet`, `DreamerAgent`, `JEPAAgent`, `VisionTransformer`, `ModularRSSM`, `create_modular_rssm` |
| **Configs** | `DreamerConfig`, `JEPAConfig`, `DiTConfig`, `DiamondConfig`, `IRISConfig` |
| **Environments** | `make_atari_env`, `GymImageEnv`, `DeepMindControlEnv`, `UnityMLAgentsEnv`, `TimeLimit`, `ActionRepeat`, etc. |
| **Operators** | `get_operator`, `DreamerOperator`, `JEPAOperator`, `IrisOperator`, `PlaNetOperator` |
| **Reward** | `RewardModel`, `ValueModel` |
| **Utilities** | `__version__` |

Example usage:

```python
from world_models import DreamerAgent, DreamerConfig
from world_models import get_operator

# Training
cfg = DreamerConfig()
cfg.env = "walker-walk"
agent = DreamerAgent(cfg)
agent.train()

# Inference preprocessing
op = get_operator('dreamer', image_size=64, action_dim=6)
processed = op.process({'image': image, 'action': action})
```

## Core Modules

- `world_models.models`: High-level models and agents
- `world_models.configs`: Configuration containers
- `world_models.training`: Script-style training entrypoints

## Environment Integration

- `world_models.envs`: DMC, Gym/Gymnasium, Atari, MuJoCo, Unity ML-Agents adapters
- `world_models.envs.wrappers`: Action repeat, normalization, time limits

## World Model Building Blocks

- `world_models.models.dreamer_rssm`: Recurrent state-space model
- `world_models.models.modular_rssm`: Modular RSSM
- `world_models.vision`: Encoders/decoders
- `world_models.reward`: Reward and value heads

## Representation Learning and Diffusion

- `world_models.models.vit`: Vision Transformer and JEPA
- `world_models.models.diffusion`: DDPM and DiT
- `world_models.masks`: Mask collators

## Data and Memory

- `world_models.datasets`: CIFAR-10, ImageNet-1K loaders
- `world_models.memory`: Replay buffers

## Utilities

- `world_models.utils`: Logging, transforms, benchmarks
- `world_models.benchmarks`: CLI and reporting
