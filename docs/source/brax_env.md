# Brax Environments

This page explains how to configure TorchWM world-model training for
[JAX/Brax](https://github.com/google/brax) continuous-control environments.

## Overview

Brax environments use a functional API: `reset(rng)` returns an environment
state, and `step(state, action)` returns the next state. TorchWM wraps this API
with `BraxImageEnv`, a Gym-like adapter that stores the latest Brax state between
calls and returns image observations in the format expected by pixel-based agents
such as Dreamer.

The adapter supports:

- Brax environment IDs such as `"ant"`, `"humanoid"`, `"hopper"`,
  `"halfcheetah"`, and `"walker2d"`.
- Pre-built Brax environment instances with `reset(rng)`, `step(state, action)`,
  and `action_size` attributes.
- Optional JIT compilation of Brax `reset` and `step` functions.
- Continuous action spaces exposed as `[-1, 1]` vectors.
- Deterministic RGB feature-band images for vector observations.
- Raw vector observations in `info["vector_observation"]` for diagnostics.

## Installation

Install TorchWM with the Brax optional extra:

```bash
pip install torchwm[brax]
```

If you are working from a local checkout, install the extra from the repository
root:

```bash
pip install -e .[brax]
```

## Dreamer configuration

Set `DreamerConfig.env_backend` to `"brax"` and choose a Brax task name:

```python :class: thebe
from torchwm import DreamerConfig
from torchwm import Dreamer

cfg = DreamerConfig()
cfg.env_backend = "brax"
cfg.env = "ant"
cfg.image_size = (64, 64)
cfg.time_limit = 1000
cfg.action_repeat = 1

# Brax-specific options.
cfg.brax_backend = "generalized"  # Use "mjx" when supported by your install.
cfg.brax_jit = True
cfg.brax_auto_reset = False
# Optionally suppress noisy optional MuJoCo/MJX Warp import messages that may
# appear during Brax imports when Warp isn't installed. These messages are
# harmless but can clutter logs during tests or normal runs. Enabled by
# default.
cfg.brax_suppress_warp_warnings = True

agent = Dreamer(cfg)
agent.train()
```

`BraxImageEnv` passes `cfg.time_limit` to Brax as the environment
`episode_length`. TorchWM's regular `TimeLimit` wrapper is still applied after
action-repeat wrapping, so keep `cfg.time_limit` aligned with the horizon you
want the world model to collect.

## Direct adapter usage

You can also construct the adapter directly:

```python :class: thebe
from torchwm import make_brax_env

env = make_brax_env(
    "ant",
    seed=0,
    size=(64, 64),
    backend="generalized",
    episode_length=1000,
    jit=True,
)

obs = env.reset()
action = env.action_space.sample()
next_obs, reward, done, info = env.step(action)

print(obs["image"].shape)                 # (3, 64, 64)
print(info["vector_observation"].shape)   # Raw Brax observation vector.
```

## Observation format

Many Brax tasks return vector observations rather than rendered camera frames.
TorchWM converts those vectors into deterministic RGB feature-band images so the
same pixel-based model code can consume the environment stream. This conversion
is intended as a compatibility layer for world-model pipelines that expect
images; use `info["vector_observation"]` when you need access to the raw Brax
state observation for debugging or custom losses.

The adapter always advertises this observation space:

```python
{
    "image": Box(low=0, high=255, shape=(3, height, width), dtype=uint8)
}
```

## Action format

Brax actions are continuous vectors. TorchWM exposes a continuous Gymnasium
`Box` action space with shape `(env.action_size,)` and bounds `[-1, 1]`. Incoming
actions are clipped to this range before being forwarded to Brax.

## Troubleshooting

### Missing Brax or JAX

If you see an import error for `brax`, `jax`, or `jax.numpy`, install the Brax
extra:

```bash
pip install torchwm[brax]
```

### Backend selection

The default Brax backend is `"generalized"`. Some Brax versions and tasks also
support `"mjx"`; set `cfg.brax_backend = "mjx"` only when your installed Brax
version supports it for the selected environment.

### JIT compilation

JIT is enabled by default through `cfg.brax_jit = True`. Disable it while
debugging shape or dtype issues:

```python
cfg.brax_jit = False
```
