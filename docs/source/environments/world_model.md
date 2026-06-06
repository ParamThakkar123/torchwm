# World Model Env

`WorldModelEnv` wraps a trained or adapter-backed world model as a Gymnasium-compliant environment. It lets you run simulated rollouts with the same API shape expected by RL libraries such as Stable-Baselines3, TorchRL, and CleanRL:

- `reset(seed=..., options=...) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `observation_space`, `action_space`, `render()`, and `close()`

The wrapper is model-agnostic. You can either expose common model methods (`env_step`, `step`, `predict_step`, `predict`, `imagine_step`, `transition`, or `__call__`) or provide explicit adapter callables.

## Basic usage

```python
import gymnasium as gym
import numpy as np

from world_models.envs import WorldModelEnv


def transition(model, state, action):
    next_state = model.imagine_step(state, action)
    obs = model.decode(next_state)
    reward = model.reward(next_state)
    done = model.termination(next_state)
    return {
        "state": next_state,
        "observation": obs,
        "reward": reward,
        "terminated": done,
    }


env = WorldModelEnv(
    trained_model,
    observation_space=gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
    action_space=gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
    initial_state=initial_latent,
    transition_fn=transition,
    max_episode_steps=50,
)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Factory and public API

The direct factory is `make_world_model_env`:

```python
from world_models.envs import make_world_model_env

env = make_world_model_env(
    trained_model,
    observation_space=obs_space,
    action_space=act_space,
    transition_fn=transition,
)
```

The top-level factory also includes a `world-model` backend:

```python
import torchwm

env = torchwm.make_env(
    trained_model,
    backend="world-model",
    observation_space=obs_space,
    action_space=act_space,
    transition_fn=transition,
)
```

Aliases include `world_model`, `model`, and `wm`.

## Accepted adapter return forms

`reset_fn` may return:

- `obs`
- `(obs, info)`
- `(state, obs)`
- `(state, obs, info)`
- a mapping with `state`, `observation`/`obs`/`image`, and optional `info`

`transition_fn` or model transition methods may return:

- `(obs, reward, terminated, truncated, info)`
- `(obs, reward, done, info)`
- `(state, obs, reward)`
- `(state, obs, reward, terminated, truncated, info)`
- a mapping with `state`/`next_state`, `observation`/`obs`/`image`, `reward`, `terminated`/`done`, `truncated`, and optional `info`

If a transition omits a reward or termination flag, pass `reward_fn` and `terminal_fn`. Missing rewards default to `0.0`; missing termination defaults to `False`. `max_episode_steps` sets `truncated=True` when the simulated rollout reaches the time limit.
