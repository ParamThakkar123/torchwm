# Isaac Lab Integration Tutorial

This tutorial shows how to use Isaac Lab environments with TorchWM world models like Dreamer.

## Prerequisites

Install Isaac Lab: Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

## Creating an Isaac Lab Environment

First, create a simple Isaac Lab environment. For this tutorial, we'll assume you have a basic task set up.

```python
import isaaclab
from isaaclab import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher()
simulation_app = app_launcher.app

# Create environment
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks import IsaacLabTask

# For this example, assume a simple task
env = ManagerBasedRLEnv(
    cfg=IsaacLabTask.Cfg(
        num_envs=32,
        env_spacing=2.0,
        action_space="continuous",
    ),
    device="cuda",
    seed=42,
)
```

## Wrapping with TorchWM

TorchWM provides `IsaacLabImageEnv` to wrap Isaac Lab environments for image-based world models.

```python
from world_models.envs.isaaclab_env import IsaacLabImageEnv

# Wrap the Isaac Lab env
torchwm_env = IsaacLabImageEnv(env=env, seed=42, size=(64, 64))
```

## Using with Dreamer

Now, use the wrapped environment with the Dreamer model.

```python
from world_models.models.dreamer import DreamerConfig, Dreamer
from world_models.training.train_dreamer import train_dreamer

# Configure Dreamer for Isaac Lab
config = DreamerConfig(
    env_name="isaaclab",  # Custom backend
    action_dim=torchwm_env.action_space.shape[0],
    obs_shape=(3, 64, 64),
    device="cuda",
    batch_size=32,
    num_envs=32,
)

# Create model
dreamer = Dreamer(config)

# Train (this is a simplified example)
# In practice, you'd use the training loop with proper data collection
for batch in data_loader:
    loss = dreamer.training_step(batch)
    # ...
```

## GPU Vectorized Environments

For better performance, use the GPU-accelerated vectorized environment.

```python
from world_models.envs.vector_env import GPUVectorizedEnv

# Factory function for Isaac Lab env
def isaaclab_factory():
    # Return your Isaac Lab env creation here
    return env  # Pre-created env

# Create vectorized env
vec_env = GPUVectorizedEnv(
    env_factory=isaaclab_factory,
    num_envs=32,
    device="cuda",
    async_batching=True,
)

# Use with Dreamer
obs = vec_env.reset_batch()
actions = torch.randn(32, action_dim, device="cuda")
next_obs, reward, done, info = vec_env.step_batch(actions)
```

## Performance Tips

- Use CUDA devices for GPU acceleration.
- Enable async batching for overlapping computation and data transfer.
- Ensure Isaac Lab environments are configured for vectorized simulation.

## Troubleshooting

- If you encounter rendering issues, ensure Isaac Sim is properly launched.
- For environment creation errors, check your Isaac Lab task configuration.
- GPU memory issues: Reduce num_envs or batch_size.