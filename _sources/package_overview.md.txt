# Package Overview

TorchWM is organized into focused modules so you can use only the pieces you need.

## Core APIs

- `world_models.models`: High-level models and agents (`Dreamer`, `DreamerAgent`, `Planet`, `JEPAAgent`)
- `world_models.configs`: Configuration containers for Dreamer, JEPA, and diffusion runs
- `world_models.training`: Script-style training entrypoints

## Environment Integration

- `world_models.envs`: Environment adapters for DMC, Gym/Gymnasium, Atari, MuJoCo, Unity ML-Agents
- `world_models.envs.wrappers`: Common wrappers for action repeat, action normalization, time limits, and observation shaping

## World Model Building Blocks

- `world_models.models.dreamer_rssm`: Recurrent state-space model used by Dreamer
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

- `world_models.utils.dreamer_utils`: Logging, parameter freezing, and TD(lambda) return computation
- `world_models.utils.jepa_utils`: Optimizer schedules, distributed helpers, and training meters
- `world_models.transforms`: Data augmentation pipelines used by JEPA/vision training

## Which API Should I Use?

- End-to-end Dreamer training: `DreamerAgent`
- End-to-end JEPA training: `JEPAAgent`
- Low-level model experimentation: `Dreamer`, `RSSM`, decoder/encoder modules
- Custom data pipelines: `make_cifar10`, `make_imagenet1k`, `make_imagefolder`
