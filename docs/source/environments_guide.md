# Environments Guide

TorchWM supports several environment backends for training, evaluation, and dataset collection. The detailed backend documentation now lives on separate pages so each environment family can document its own installation requirements, factory functions, observation/action contracts, and troubleshooting notes.

```{toctree}
:maxdepth: 1

Environment backend overview <environments/index>
DeepMind Control Suite <environments/dmc>
Gym and Gymnasium <environments/gym>
Atari <environments/atari>
MuJoCo <environments/mujoco>
Unity ML-Agents <environments/unity>
Vectorized Environments <environments/vectorized>
Wrappers <environments/wrappers>
```

## Quick start

Use `DreamerConfig.env_backend` for Dreamer-compatible DMC, Gym/Gymnasium, MuJoCo, and Unity environments:

```python :class: thebe
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "dmc"  # "dmc", "gym", "mujoco", or "unity_mlagents"
cfg.env = "walker-walk"
cfg.image_size = 64
cfg.action_repeat = 2
cfg.time_limit = 1000
```

For direct environment construction, import the relevant factory from `world_models.envs`:

```python :class: thebe
from world_models.envs import DeepMindControlEnv, make_gym_env, make_atari_env

dmc_env = DeepMindControlEnv("cheetah-run", seed=0, size=(64, 64))
gym_env = make_gym_env("Pendulum-v1", seed=0, size=(64, 64))
atari_env = make_atari_env("ALE/Pong-v5", obs_type="rgb", frameskip=4)
```

## Backend summary

| Backend | Page | Use when |
| --- | --- | --- |
| DeepMind Control Suite | [DMC](environments/dmc.md) | You want Dreamer-style continuous-control tasks with rendered images and native DMC state observations. |
| Gym/Gymnasium | [Gym](environments/gym.md) | You want classic control, Box2D, custom Gym environments, or generic rendered tasks converted to TorchWM image observations. |
| Atari | [Atari](environments/atari.md) | You want Atari environments through Gymnasium/ALE, native ALE vectorization, or Atari-specific DIAMOND-style preprocessing. |
| MuJoCo | [MuJoCo](environments/mujoco.md) | You want configurable Humanoid or HalfCheetah Gymnasium factories. |
| Unity ML-Agents | [Unity](environments/unity.md) | You want to train against external Unity executables with continuous-control behaviors. |
| Vectorized environments | [Vectorized](environments/vectorized.md) | You need batched rollout collection across native ALE vector envs or multiprocessing Gym-like env factories. |
| Wrappers | [Wrappers](environments/wrappers.md) | You need action repeat, time limits, action normalization, one-hot actions, reward observations, or image transforms. |

## Shared TorchWM environment conventions

- Image-based training code generally expects an observation dictionary with an `image` entry.
- DMC, Gym/Gymnasium, and Unity adapters return channel-first images shaped `(3, H, W)` with dtype `uint8`.
- Atari can expose raw ALE observations, native vectorized observations, or DIAMOND-style preprocessed frames; check the Atari page before feeding observations directly to a model.
- Dreamer environment creation applies `ActionRepeat`, `NormalizeActions`, and `TimeLimit` after constructing the selected backend.
- Use `torchwm envs list` to inspect the lightweight backend catalog available to the CLI.

## Common issues

- **Missing optional dependency**: install the backend-specific package listed on the backend page.
- **Observation shape mismatch**: verify whether your selected factory returns CHW image dictionaries, HWC images, RAM, or vector observations.
- **Action mismatch**: distinguish raw discrete action indices from one-hot vectors and normalized continuous actions.
- **Slow environments**: disable rendering where possible, use action repeat, and consider vectorized rollout collection.
