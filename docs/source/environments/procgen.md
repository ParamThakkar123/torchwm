# Procgen

The Procgen backend adapts the `procgen.ProcgenEnv` vector API to TorchWM's single-environment image interface for procedurally generated benchmark games such as CoinRun, Maze, Heist, and StarPilot.

## Install

Procgen is an optional dependency:

```bash
pip install torchwm[procgen]
```

You can also install it directly with `pip install procgen` when working from a source checkout.

## Main APIs

```python
from world_models.envs.procgen_env import make_procgen_env, list_procgen_envs

env = make_procgen_env("coinrun", seed=0, size=(64, 64))
obs = env.reset()
print(obs["image"].shape)  # (3, 64, 64)
```

The factory accepts Procgen shorthand names and Gym-style ids:

```python
make_procgen_env("coinrun")
make_procgen_env("procgen-coinrun-v0")
make_procgen_env("procgen:procgen-coinrun-v0")
```

Use `list_procgen_envs()` to inspect the supported game names.

## Dreamer configuration

```python
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "procgen"
cfg.env = "coinrun"
cfg.image_size = 64
cfg.procgen_distribution_mode = "easy"
cfg.procgen_num_levels = 0      # 0 means unlimited levels in Procgen
cfg.procgen_start_level = None  # defaults to cfg.seed
```

`env_backend` may be either `"procgen"` or `"coinrun"`. Dreamer applies the same `ActionRepeat`, `NormalizeActions`, and `TimeLimit` wrapper stack used by other online environment backends.

## Observations and actions

`ProcgenImageEnv` unwraps the leading vector dimension from `ProcgenEnv(num_envs=1)` and returns:

```python
{"image": uint8 array with shape (3, H, W)}
```

Procgen actions are discrete. For consistency with TorchWM's other discrete image adapters, the wrapper exposes a continuous one-hot-like action space with shape `(n,)` and values in `[-1, 1]`. The selected action is the index of the largest value in the model action vector.

## Available games

TorchWM recognizes the standard Procgen games: `bigfish`, `bossfight`, `caveflyer`, `chaser`, `climber`, `coinrun`, `dodgeball`, `fruitbot`, `heist`, `jumper`, `leaper`, `maze`, `miner`, `ninja`, `plunder`, and `starpilot`.
