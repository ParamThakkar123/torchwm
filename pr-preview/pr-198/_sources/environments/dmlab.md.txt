# DeepMind Lab

TorchWM supports DeepMind Lab through `DMLabEnv`, a lightweight adapter around the
native `deepmind_lab` Python module. It converts Lab RGB observations into the
channel-first image dictionary used by Dreamer and other pixel-based world-model
code.

## Installation

DeepMind Lab includes native game-engine components and is distributed separately from TorchWM. Install or build a package that exposes the `deepmind_lab` Python module before using this adapter. One option is the external `dmlab-gym` build helper:

```bash
pip install dmlab-gym
dmlab-gym build
```

If you already have a working `deepmind_lab` installation, no additional runtime package is required by the adapter.

## Dreamer configuration

```python
from world_models.configs.dreamer_config import DreamerConfig
from world_models.models.dreamer import DreamerAgent

cfg = DreamerConfig()
cfg.env_backend = "dmlab"
cfg.env = "rooms_collect_good_objects_train"
cfg.image_size = (64, 64)
cfg.dmlab_action_repeat = 4

agent = DreamerAgent(cfg)
agent.train()
```

`DMLabEnv` asks Lab for `RGB_INTERLEAVED` observations and returns them as
`obs["image"]` with shape `(3, H, W)` and dtype `uint8`. Extra observation names
can be requested with `cfg.dmlab_observations`; they are copied into the returned
observation dictionary under their native DeepMind Lab names.

## Direct usage

```python
from world_models.envs import make_dmlab_env

env = make_dmlab_env("rooms_collect_good_objects_train", seed=0, size=(64, 64))
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

The default action space is a normalized one-hot `Box[-1, 1]` over a compact set
of navigation actions. You can pass `action_set=` to `DMLabEnv` or set
`cfg.dmlab_action_set` to use a custom list of native seven-element Lab actions.

## Backend options

| Option | Default | Description |
| --- | --- | --- |
| `cfg.dmlab_action_repeat` | `4` | Native Lab frame repeat passed to `env.step(..., num_steps=...)`. |
| `cfg.dmlab_action_set` | `None` | Optional custom 2D array of native Lab actions. |
| `cfg.dmlab_observations` | `None` | Additional native observation names. `RGB_INTERLEAVED` is always included. |
| `cfg.dmlab_config` | `None` | Extra Lab config values. Width and height are derived from `cfg.image_size`. |
| `cfg.dmlab_renderer` | `"hardware"` | Renderer argument forwarded to `deepmind_lab.Lab`. |

TorchWM's shared Dreamer wrapper stack still applies `cfg.action_repeat` outside
the DMLab adapter. If you only want native Lab frame repeat, leave
`cfg.action_repeat = 1`.
