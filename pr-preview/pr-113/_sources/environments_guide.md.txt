# Environments Guide

This guide covers how to set up and use different environments with TorchWM.

## Supported Environments

TorchWM supports multiple environment backends for training and evaluation.

## DeepMind Control Suite (DMC)

The DeepMind Control Suite provides high-quality continuous control tasks.

### Setup

```bash
pip install dm-control
```

### Configuration

```python :class: thebe
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "dmc"
cfg.env = "walker-walk"  # or "cartpole-balance", "finger-turn_hard", etc.
```

### Available Tasks

- **Locomotion**: `walker-walk`, `walker-run`, `cheetah-run`
- **Manipulation**: `finger-turn_hard`, `finger-turn_easy`
- **Balance**: `cartpole-balance`, `cartpole-swingup`
- **Others**: `ball_in_cup-catch`, `point_mass-easy`

## Gym/Gymnasium

Standard reinforcement learning environments.

### Setup

```bash
pip install gymnasium
pip install gymnasium[classic-control,atari,box2d]  # optional extras
```

### Configuration

```python :class: thebe
cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"
cfg.gym_render_mode = "rgb_array"  # for rendering
```

### Custom Environments

```python :class: thebe
import gymnasium as gym
from world_models.envs import GymWrapper

# Use existing gym env
env = gym.make("CartPole-v1")

# Or wrap custom env
class MyEnv(gym.Env):
    # Implement gym.Env interface
    pass

cfg.env_instance = MyEnv()
```

## Unity ML-Agents

For complex 3D environments and simulations.

### Setup

1. Install Unity ML-Agents: https://github.com/Unity-Technologies/ml-agents
2. Build your environment executable

### Configuration

```python :class: thebe
cfg = DreamerConfig()
cfg.env_backend = "unity_mlagents"
cfg.unity_file_name = r"C:\Path\To\Env.exe"
cfg.unity_behavior_name = "YourBehavior"
cfg.unity_no_graphics = True  # faster training
cfg.unity_time_scale = 20.0  # speed up simulation
```

## Custom Environments

Implement your own environments using the TorchWM interface.

### Environment Interface

```python :class: thebe
from world_models.envs.base import BaseEnvironment

class MyEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your environment

    def reset(self):
        # Reset environment
        return initial_observation

    def step(self, action):
        # Execute action
        # Return: observation, reward, done, info
        return obs, reward, done, {}

    def render(self):
        # Optional rendering
        pass

    def close(self):
        # Cleanup
        pass
```

### Registration

```python :class: thebe
from world_models.envs import register_environment

register_environment("my_env", MyEnvironment)

# Then use in config
cfg.env_backend = "custom"
cfg.env = "my_env"
```

## Environment Wrappers

Apply transformations to environments:

```python :class: thebe
from world_models.envs.wrappers import (
    FrameStackWrapper,
    ActionRepeatWrapper,
    RewardScaleWrapper
)

# Stack frames
env = FrameStackWrapper(env, num_stack=4)

# Repeat actions
env = ActionRepeatWrapper(env, repeat=4)

# Scale rewards
env = RewardScaleWrapper(env, scale=0.1)
```

## Multi-Environment Training

Train on multiple environments simultaneously:

```python :class: thebe
cfg = DreamerConfig()
cfg.envs = ["walker-walk", "cheetah-run", "finger-turn_hard"]
cfg.env_sampling = "uniform"  # or "weighted"
```

## Evaluation Environments

Use separate environments for evaluation:

```python :class: thebe
cfg = DreamerConfig()
cfg.eval_env = "walker-run"  # different from training env
cfg.eval_episodes = 10
cfg.eval_freq = 10000
```

## Debugging Environments

### Visualization

```python :class: thebe
from world_models.envs.utils import play_environment

# Play environment interactively
play_environment(cfg, num_episodes=5)
```

### Recording

```python :class: thebe
from world_models.envs.utils import record_environment

# Record episodes
record_environment(cfg, num_episodes=3, save_path="videos/")
```

## Performance Tips

### Environment Speed

1. **Use vectorized environments** for faster training
2. **Disable rendering** during training
3. **Use action repeat** to reduce environment calls
4. **Batch observations** when possible

### Memory Usage

1. **Limit frame buffers** in memory
2. **Use efficient observation types** (avoid large images if possible)
3. **Clear environment caches** periodically

## Common Issues

### Import Errors
- Ensure all dependencies are installed
- Check environment paths for Unity ML-Agents

### Performance Issues
- Profile environment step times
- Check for unnecessary computations in step()

### Observation Issues
- Verify observation shapes match model inputs
- Check normalization ranges

### Action Issues
- Ensure action spaces match agent outputs
- Validate action bounds and types