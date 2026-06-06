# Package Overview

TorchWM is organized into focused modules so you can use only the pieces you need.

## Quick Import (Public API)

For applications and examples, prefer the installed package name, `torchwm`. It
mirrors the TorchWM implementation package and exposes the same lazy public
API without importing optional training backends until you use them.

```python
import torchwm

print(torchwm.list_models())
agent = torchwm.create_model("dreamer", env="walker-walk", total_steps=1_000_000)
env = torchwm.make_env("CartPole-v1", backend="gym")
op = torchwm.get_operator("dreamer", image_size=64, action_dim=6)
```

Use `torchwm` for direct component imports as well as factory helpers:

```python
from torchwm import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env = "walker-walk"
agent = DreamerAgent(cfg)
```

### Available Exports

| Category | Exports |
|----------|--------|
| **Friendly factories** | `create_config`, `create_model`, `make_env`, `list_models`, `list_env_backends`, `list_envs` |
| **Models** | `Dreamer`, `Planet`, `DreamerAgent`, `JEPAAgent`, `IRISAgent`, `Genie`, `VisionTransformer`, `ModularRSSM`, `create_modular_rssm` |
| **Configs** | `DreamerConfig`, `JEPAConfig`, `DiTConfig`, `DiamondConfig`, `IRISConfig`, `GenieConfig`, `GenieSmallConfig` |
| **Environments** | `make_atari_env`, `make_gym_env`, `make_mujoco_env`, `make_robotics_env`, `make_brax_env`, `GymImageEnv`, wrappers, etc. |
| **Operators** | `get_operator`, `DreamerOperator`, `JEPAOperator`, `IrisOperator`, `PlaNetOperator` |
| **Reward** | `RewardModel`, `ValueModel` |
| **Utilities** | `__version__` |

Example usage:

```python
import torchwm

# Training
agent = torchwm.create_model("dreamer", env="walker-walk", total_steps=1_000_000)
agent.train()

# Inference preprocessing
op = torchwm.get_operator("dreamer", image_size=64, action_dim=6)
processed = op.process({"image": image, "action": action})
```

## Core Modules

- `world_models.models`: High-level models and agents (`Dreamer`, `DreamerAgent`, `Planet`, `JEPAAgent`)
- `world_models.configs`: Configuration containers for Dreamer, JEPA, and diffusion runs
- `world_models.training`: Script-style training entrypoints

## Environment Integration

- `world_models.envs`: DMC, Gym/Gymnasium, Atari, MuJoCo, Unity ML-Agents adapters
- `world_models.envs.wrappers`: Action repeat, normalization, time limits

## World Model Building Blocks

- `world_models.models.dreamer_rssm`: Recurrent state-space model used by Dreamer
- `world_models.models.modular_rssm`: Modular RSSM with swappable encoder/decoder/backbone for research experiments
- `world_models.vision`: Encoders/decoders and action heads for latent dynamics models
- `world_models.reward`: Reward and value prediction heads
- `world_models.observations`: Symbolic and visual observation reconstruction modules

## Representation Learning and Diffusion

- `world_models.models.vit`: Vision Transformer and JEPA predictor components
- `world_models.models.diffusion`: DDPM scheduler and DiT model implementation
- `world_models.masks`: Mask collators for JEPA-style context/target masking

## Data and Memory

- `world_models.datasets`: CIFAR-10, ImageNet-1K, and generic `ImageFolder` dataset loaders
- `world_models.memory`: Replay buffers for Dreamer and episode-based memory for PlaNet/RSSM

## Utilities

- `world_models.utils`: Logging, parameter freezing, transforms
- `world_models.transforms`: Data augmentation pipelines
- `world_models.benchmarks`: CLI and reporting utilities

## Which API Should I Use?

- End-to-end Dreamer training: `DreamerAgent`
- End-to-end JEPA training: `JEPAAgent`
- Low-level model experimentation: `Dreamer`, `RSSM`, decoder/encoder modules
- Custom world model architectures: `ModularRSSM` with swappable encoder/decoder/backbone
- Custom data pipelines: `make_cifar10`, `make_imagenet1k`, `make_imagefolder`
