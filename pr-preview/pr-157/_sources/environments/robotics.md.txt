# Gymnasium Robotics

TorchWM supports Gymnasium Robotics environments through shared factories and environment catalogs for online world models. The adapter imports and registers `gymnasium_robotics` before calling `gymnasium.make`, which exposes every environment registered by the installed Gymnasium Robotics package, including the legacy MuJoCo v2/v3 task ids that Gymnasium moved out of the core package.

## Install

```bash
pip install "torchwm[robotics]"
```

You can also install the package directly:

```bash
pip install gymnasium-robotics
```

## World-model availability

Gymnasium Robotics ids are listed dynamically from Gymnasium's registry after `gymnasium_robotics` is installed. The catalog exposes those ids to the online world-model families (Dreamer, PlaNet/RSSM, IRIS, DIAMOND, Genie, and DiT). I-JEPA/JEPA is intentionally excluded because it trains on image datasets rather than online Gymnasium environments.

## Dreamer configuration

```python
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "robotics"
cfg.env = "HalfCheetah-v2"
cfg.gym_render_mode = "rgb_array"
```

The Dreamer wrapper stack still applies action repeat, action normalization, and time limits after the Robotics environment is converted to TorchWM image observations.

## Factory usage

```python
from world_models.envs.robotics_env import make_robotics_env

env = make_robotics_env(
    "HalfCheetah-v2",
    seed=0,
    size=(64, 64),
    render_mode="rgb_array",
)
```

`make_robotics_env()` returns a `GymImageEnv`, so observations follow TorchWM's image contract: `{"image": uint8[3, H, W]}`. Lower-level helpers also list all installed Robotics ids and retry Gymnasium creation when a moved MuJoCo v2/v3 id requires Robotics registration.

## MuJoCo compatibility

The MuJoCo and generic Gym image factories also recognize the moved-environment error emitted by Gymnasium. If a legacy id such as `HalfCheetah-v2` fails because Gymnasium says the v2/v3 MuJoCo environments moved to Gymnasium Robotics, TorchWM imports/registers `gymnasium_robotics` and retries automatically.

## Troubleshooting

- **Moved v2/v3 MuJoCo environments**: install `torchwm[robotics]` or `gymnasium-robotics`.
- **Renderer errors**: keep `render_mode="rgb_array"` for image-based Dreamer runs and verify that the underlying Gymnasium Robotics environment can render locally.
- **Newer Gymnasium ids**: modern ids such as `HalfCheetah-v5` can continue to use `env_backend="mujoco"` or `env_backend="gym"` when they are available in your Gymnasium installation.
