# Environment Backends

TorchWM ships environment adapters for pixel-based world-model training, model-based reinforcement learning, and benchmark collection. Each backend page explains the installation requirements, factory functions, observation and action conventions, configuration fields, and common troubleshooting steps for that backend.

```{toctree}
:maxdepth: 1

DeepMind Control Suite <dmc>
Gym and Gymnasium <gym>
Atari <atari>
MuJoCo <mujoco>
Unity ML-Agents <unity>
Vectorized Environments <vectorized>
Wrappers <wrappers>
```

## Choosing a backend

| Backend | Best for | Primary APIs | Typical observations | Typical actions |
| --- | --- | --- | --- | --- |
| [DeepMind Control Suite](dmc.md) | Dreamer-style continuous-control tasks with state and rendered image observations | `DeepMindControlEnv`, `env_backend="dmc"` | Dict with DMC state keys plus `image` | Continuous `Box` from the DMC action spec |
| [Gym and Gymnasium](gym.md) | Classic control, Box2D, custom Gym environments, and generic rendered tasks | `GymImageEnv`, `make_gym_env`, `env_backend="gym"` | Dict with `image` only | Original continuous `Box` or one-hot vector for discrete actions |
| [Atari](atari.md) | Atari 2600 environments through Gymnasium/ALE, with optional DIAMOND-style preprocessing documented on the Atari page | `make_atari_env`, `make_atari_vector_env`, `make_diamond_atari_env` | ALE RGB/RAM observations or preprocessed resized RGB frames | Discrete Atari actions |
| [MuJoCo](mujoco.md) | Parameterized Humanoid and HalfCheetah factories | `make_humanoid_env`, `make_half_cheetah_env` | Gymnasium MuJoCo observations | Continuous `Box` |
| [Unity ML-Agents](unity.md) | External Unity executable simulations with continuous-control behaviors | `UnityMLAgentsEnv`, `env_backend="unity_mlagents"` | Dict with `image` | Continuous `Box[-1, 1]` |
| [Vectorized environments](vectorized.md) | Multiprocess/vector rollout collection and native ALE vectorization | `TorchVectorizedEnv`, `make_atari_vector_env` | Batched observations | Batched actions |
| [Wrappers](wrappers.md) | Shared preprocessing, action conversion, time limits, reward observations, and image transforms | `world_models.envs.wrappers` | Backend-dependent | Backend-dependent |

## Shared conventions

Most TorchWM training code expects image observations as a dictionary entry named `image` with channel-first shape `(3, H, W)` and `uint8` values. Backend adapters that wrap vector-only environments synthesize an image representation so pixel-based agents can still run.

Dreamer environment construction applies a standard wrapper stack after creating DMC, Gym/Gymnasium, or Unity environments:

1. `ActionRepeat` repeats each selected action for `cfg.action_repeat` environment steps.
2. `NormalizeActions` exposes finite continuous action bounds as normalized `[-1, 1]` policy outputs.
3. `TimeLimit` truncates episodes after `cfg.time_limit // cfg.action_repeat` wrapper steps.

Use `torchwm envs list` to print the lightweight backend catalog used by the CLI.
