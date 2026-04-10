# Brax Integration Tutorial

This tutorial demonstrates how to integrate Brax environments with TorchWM for high-performance world model training.

## Prerequisites

Install Brax: `pip install brax`

## Creating a Brax Environment

Brax provides fast JAX-based physics simulation.

```python
import brax.envs
from brax import envs

# Create a Brax environment
env = envs.get_environment('ant')  # Or brax.envs.ant.Ant()
```

## Using GPUVectorizedEnv with Brax

TorchWM's GPUVectorizedEnv automatically detects Brax environments and applies JAX JIT compilation for performance.

```python
from world_models.envs.vector_env import GPUVectorizedEnv

# Factory function
def brax_factory():
    return brax.envs.ant.Ant()

# Create vectorized GPU environment
vec_env = GPUVectorizedEnv(
    env_factory=brax_factory,
    num_envs=32,
    device="cuda",  # PyTorch device, but JAX will use its own
    async_batching=True,
)

print("JAX JIT applied to Brax environment for improved performance.")
```

## Training with Dreamer

Use the vectorized environment with Dreamer.

```python
import torch
from world_models.models.dreamer import DreamerConfig, Dreamer

# Configure for Brax Ant
config = DreamerConfig(
    env_name="brax",
    action_dim=vec_env.action_space.shape[0],  # 8 for Ant
    obs_shape=(3, 64, 64),  # Image observations
    device="cuda",
    batch_size=32,
    num_envs=32,
)

dreamer = Dreamer(config)

# Training loop (simplified)
for episode in range(1000):
    obs = vec_env.reset_batch()
    episode_reward = 0
    
    for step in range(1000):
        # Sample actions
        actions = torch.randn(config.num_envs, config.action_dim, device=config.device)
        
        # Step environment
        result = vec_env.step_batch(actions)
        next_obs = result["obs"]
        reward = result["reward"]
        done = result["done"]
        
        # Add to replay buffer
        # ... (implementation depends on your buffer)
        
        # Train dreamer
        if len(buffer) > config.batch_size:
            batch = buffer.sample()
            loss = dreamer.training_step(batch)
            print(f"Loss: {loss}")
        
        obs = next_obs
        episode_reward += reward.mean().item()
        
        if done.any():
            break
    
    print(f"Episode {episode}: Reward {episode_reward}")
```

## Performance Optimization

- JAX JIT compilation is automatically applied to Brax envs.
- Use GPU devices for data transfer and PyTorch operations.
- Brax runs on CPU/JAX, so GPU benefits come from PyTorch side.

## Custom Brax Environments

You can use any Brax environment:

```python
# Different environments
def hopper_factory():
    return brax.envs.hopper.Hopper()

def humanoid_factory():
    return brax.envs.humanoid.Humanoid()

# All work the same way
vec_env = GPUVectorizedEnv(env_factory=humanoid_factory, num_envs=16)
```

## Troubleshooting

- JAX memory issues: Reduce num_envs.
- JIT compilation fails: Check Brax version compatibility.
- Array conversion errors: Ensure JAX and NumPy versions are compatible.