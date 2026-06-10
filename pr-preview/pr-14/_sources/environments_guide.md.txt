# Environments Guide

TorchWM supports several environment backends for training, evaluation, and dataset collection. The detailed backend documentation now lives on separate pages so each environment family can document its own installation requirements, factory functions, observation/action contracts, and troubleshooting notes.

```{toctree}
:maxdepth: 1

Environment backend overview <environments/index>
DeepMind Control Suite <environments/dmc>
DeepMind Lab <environments/dmlab>
Gym and Gymnasium <environments/gym>
Brax <environments/brax>
Atari <environments/atari>
Procgen <environments/procgen>
MuJoCo <environments/mujoco>
Gymnasium Robotics <environments/robotics>
Unity ML-Agents <environments/unity>
Vectorized Environments <environments/vectorized>
Wrappers <environments/wrappers>
```

## Quick start

Use `DreamerConfig.env_backend` for Dreamer-compatible DMC, DMLab, Gym/Gymnasium, MuJoCo, Gymnasium Robotics, Procgen, BSuite, Brax, and Unity environments. Choose the backend that matches your installed optional dependencies and task source.

```python :class: thebe
from torchwm import DreamerConfig

cfg = DreamerConfig()
# env_backend may be one of: "dmc", "dmlab", "gym", "mujoco", "robotics", "procgen", "bsuite", "brax", or "unity_mlagents"
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.image_size = 64
cfg.action_repeat = 2
cfg.time_limit = 1000
```

For direct environment construction, use the top-level `torchwm` factories:

```python :class: thebe
from torchwm import (
    DeepMindControlEnv,
    make_atari_env,
    make_brax_env,
    make_bsuite_env,
    make_dmlab_env,
    make_gym_env,
    make_procgen_env,
    make_robotics_env,
)

dmc_env = DeepMindControlEnv("cheetah-run", seed=0, size=(64, 64))
dmlab_env = make_dmlab_env("rooms_collect_good_objects_train", seed=0, size=(64, 64))
gym_env = make_gym_env("Pendulum-v1", seed=0, size=(64, 64))
brax_env = make_brax_env("ant", seed=0, image_size=(64, 64))
bsuite_env = make_bsuite_env("catch/0", seed=0, size=(64, 64))
atari_env = make_atari_env("ALE/Pong-v5", obs_type="rgb", frameskip=4)
procgen_env = make_procgen_env("coinrun", seed=0, size=(64, 64))
robotics_env = make_robotics_env("HalfCheetah-v2", seed=0, size=(64, 64))
```

## Backend summary

| Backend | Page | Use when |
| --- | --- | --- |
| DeepMind Control Suite | [DMC](environments/dmc.md) | You want Dreamer-style continuous-control tasks with rendered images and native DMC state observations. |
| DeepMind Lab | [DMLab](environments/dmlab.md) | You want 3D navigation and puzzle tasks from DeepMind Lab with image observations and compact discrete actions. |
| DeepMind BSuite | `BSuiteImageEnv`, `env_backend="bsuite"` | You want small diagnostic RL benchmark tasks such as `catch/0` or `deep_sea/0`. |
| Gym/Gymnasium | [Gym](environments/gym.md) | You want classic control, Box2D, custom Gym environments, or generic rendered tasks converted to TorchWM image observations. |
| Brax | [Brax](environments/brax.md) | You want JAX/Brax continuous-control tasks wrapped in a Gym-like image adapter for TorchWM training loops. |
| Atari | [Atari](environments/atari.md) | You want Atari environments through Gymnasium/ALE, native ALE vectorization, or Atari-specific DIAMOND-style preprocessing. |
| Procgen | [Procgen](environments/procgen.md) | You want procedurally generated 2D benchmark games with image observations. |
| MuJoCo | [MuJoCo](environments/mujoco.md) | You want Gymnasium MuJoCo task ids or native MJCF/MJB models. |
| Gymnasium Robotics | [Gymnasium Robotics](environments/robotics.md) | You need any id registered by Gymnasium Robotics, including legacy MuJoCo v2/v3 ids. |
| Unity ML-Agents | [Unity](environments/unity.md) | You want to train against external Unity executables with continuous-control behaviors. |
| Vectorized environments | [Vectorized](environments/vectorized.md) | You need batched rollout collection across native ALE vector envs or multiprocessing Gym-like env factories. |
| Wrappers | [Wrappers](environments/wrappers.md) | You need action repeat, time limits, action normalization, one-hot actions, reward observations, or image transforms. |

## Shared TorchWM environment conventions

- Image-based training code generally expects an observation dictionary with an `image` entry.
- DMC, DMLab, Gym/Gymnasium, MuJoCo, Gymnasium Robotics, Procgen, and Unity adapters return channel-first images shaped `(3, H, W)` with dtype `uint8`.
- Atari can expose raw ALE observations, native vectorized observations, or DIAMOND-style preprocessed frames; check the Atari page before feeding observations directly to a model.
- Dreamer environment creation applies `ActionRepeat`, `NormalizeActions`, and `TimeLimit` after constructing the selected backend. The lightweight catalog exposes Gymnasium Robotics ids to online world-model families except I-JEPA/JEPA, which uses image datasets rather than online Gymnasium environments.
- Use `torchwm envs list` to inspect the lightweight backend catalog available to the CLI.

## Common issues

- **Missing optional dependency**: install the backend-specific package listed on the backend page.
- **Observation shape mismatch**: verify whether your selected factory returns CHW image dictionaries, HWC images, RAM, or vector observations.
- **Action mismatch**: distinguish raw discrete action indices from one-hot vectors and normalized continuous actions.
- **Brax backend mismatch**: set `cfg.brax_backend` only to backends supported by your installed Brax release (for example `"generalized"` or `"mjx"`).
- **Slow environments**: disable rendering where possible, use action repeat, and consider vectorized rollout collection.
