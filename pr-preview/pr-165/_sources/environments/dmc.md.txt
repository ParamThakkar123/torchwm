# DeepMind Control Suite

The DeepMind Control Suite (DMC) backend is the default Dreamer environment path in TorchWM. It wraps `dm_control.suite` tasks with a Gym-like interface, keeps all native DMC state observations, and adds a rendered RGB image so image-based world models can train on a consistent observation contract.

## Install

```bash
pip install dm-control
```

DMC is not part of TorchWM's minimal dependencies. Install it in the Python environment that runs training or evaluation.

## Main API

```python
from world_models.envs.dmc import DeepMindControlEnv

env = DeepMindControlEnv("cheetah-run", seed=0, size=(64, 64))
obs = env.reset()
```

The environment name uses a `domain-task` string. TorchWM splits the string at the first hyphen. For example, `cheetah-run` maps to `domain="cheetah"` and `task="run"`. The special shorthand `cup-*` maps to DMC's `ball_in_cup` domain.

## Dreamer configuration

```python
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.seed = 0
cfg.image_size = 64
cfg.action_repeat = 2
cfg.time_limit = 1000
```

`world_models.models.dreamer.make_env()` recognizes `env_backend="dmc"`, creates `DeepMindControlEnv`, and then applies `ActionRepeat`, `NormalizeActions`, and `TimeLimit`.

## Common task IDs

| Category | Examples |
| --- | --- |
| Balance | `cartpole-balance`, `cartpole-swingup` |
| Locomotion | `cheetah-run`, `walker-walk`, `walker-run`, `quadruped-walk` |
| Manipulation | `finger-spin`, `finger-turn_easy`, `finger-turn_hard`, `reacher-easy` |
| Catching | `cup-catch` |

The environment catalog includes the canonical Dreamer examples: `cartpole-balance`, `cartpole-swingup`, `cheetah-run`, `finger-spin`, `reacher-easy`, `walker-walk`, `walker-run`, and `quadruped-walk`.

## Observation contract

`DeepMindControlEnv.reset()` returns a dictionary containing:

- Every key from `dm_control`'s `observation_spec()` as a `float32` Gymnasium `Box`.
- An additional `image` key with shape `(3, H, W)` and dtype `uint8`.

The image is rendered from DMC physics with `physics.render(height, width, camera_id=...)`, transposed from HWC to CHW, and copied so downstream code can store it safely.

## Action contract

The action space is a Gymnasium `Box` built from DMC's action spec minimum and maximum arrays. Dreamer creation wraps the backend in `NormalizeActions`, so policy code can emit normalized actions while the wrapper maps finite bounds back to the native DMC range.

## Cameras and rendering

Pass `camera=<id>` when constructing `DeepMindControlEnv` directly. If no camera is provided, TorchWM uses camera `2` for `quadruped` and camera `0` for other domains. Only `rgb_array` rendering is supported.

## Troubleshooting

- **`ModuleNotFoundError: dm_control`**: install `dm-control` in the active environment.
- **Task name parsing errors**: use `domain-task` format, such as `walker-walk`; use `cup-catch` for `ball_in_cup/catch`.
- **Unexpected image size**: set `cfg.image_size` or pass `size=(height, width)` directly.
- **Action range issues**: if you bypass Dreamer `make_env()`, add `NormalizeActions` yourself when the policy emits normalized actions.
