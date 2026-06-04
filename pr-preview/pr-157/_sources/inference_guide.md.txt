# Inference Guide

This guide covers how to use trained TorchWM models for inference and deployment.

## Overview

TorchWM provides standardized inference through operators and future pipelines.

## Loading Trained Models

```python :class: thebe
from world_models.models import DreamerAgent

# Load from checkpoint
agent = DreamerAgent.from_pretrained("path/to/checkpoint")
agent.eval()
```

## Using Operators for Preprocessing

See {doc}`operators_guide` for detailed operator usage.

## Basic Inference

### Dreamer

```python :class: thebe
import torch
from world_models.inference.operators import DreamerOperator

op = DreamerOperator()
agent = DreamerAgent.from_pretrained("dreamer_checkpoint")

# Single step prediction
obs = torch.randn(3, 64, 64)
action = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

with torch.no_grad():
    processed = op({'image': obs, 'action': action})
    next_obs, reward = agent.predict(processed)
```

### JEPA

```python :class: thebe
from world_models.models import JEPAAgent
from world_models.inference.operators import JEPAOperator

op = JEPAOperator()
agent = JEPAAgent.from_pretrained("jepa_checkpoint")

# Representation learning
images = [torch.randn(3, 224, 224) for _ in range(8)]
processed = op({'images': images})

with torch.no_grad():
    representations = agent.encode(processed)
```

## Rollout and Imagination

Generate imagined trajectories:

```python :class: thebe
# Dreamer imagination
from world_models.models import DreamerAgent

agent = DreamerAgent.from_pretrained("dreamer_checkpoint")

initial_obs = torch.randn(3, 64, 64)
horizon = 10

imagined_trajectory = agent.imagine_rollout(initial_obs, horizon)
# Returns dict with imagined observations, actions, rewards
```

## Batch Inference

Process multiple inputs efficiently:

```python :class: thebe
batch_size = 32
obs_batch = torch.randn(batch_size, 3, 64, 64)
action_batch = torch.randn(batch_size, 6)

processed = op({'image': obs_batch, 'action': action_batch})

with torch.no_grad():
    predictions = agent.predict_batch(processed)
```

## GPU Acceleration

Move to GPU for faster inference:

```python :class: thebe
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = agent.to(device)
processed = {k: v.to(device) for k, v in processed.items()}

with torch.no_grad():
    output = agent.predict(processed)
```

## Real-time Inference

For interactive applications:

```python :class: thebe
class InferenceServer:
    def __init__(self):
        self.agent = DreamerAgent.from_pretrained("checkpoint").eval()
        self.op = DreamerOperator()

    def predict(self, obs, action):
        processed = self.op({'image': obs, 'action': action})
        with torch.no_grad():
            return self.agent.predict(processed)

server = InferenceServer()
```

## Performance Optimization

### JIT Compilation

```python :class: thebe
from world_models.utils.jit_utils import jit_compile_module

agent = jit_compile_module(agent)
```

### Memory Efficient Inference

```python :class: thebe
from world_models.utils.memory_utils import optimize_memory_efficient_ops

optimize_memory_efficient_ops()
```

## Exporting Models

Export to ONNX or TorchScript:

```python :class: thebe
# TorchScript
scripted = torch.jit.script(agent)
torch.jit.save(scripted, "model.pt")

# ONNX
dummy_input = op({'image': torch.randn(1, 3, 64, 64), 'action': torch.randn(1, 6)})
torch.onnx.export(agent, dummy_input, "model.onnx")
```

## Integration Examples

### With Gym Environments

```python :class: thebe
import gymnasium as gym

env = gym.make("Pendulum-v1")
agent = DreamerAgent.from_pretrained("pendulum_checkpoint")
op = DreamerOperator()

obs, _ = env.reset()
done = False

while not done:
    action = agent.act(obs)  # Get action from agent
    obs, reward, done, _, _ = env.step(action)
```

### With Custom Environments

```python :class: thebe
class CustomEnv:
    def step(self, action):
        # Your environment logic
        return obs, reward, done

env = CustomEnv()
agent = DreamerAgent.from_pretrained("custom_checkpoint")

for episode in range(10):
    obs = env.reset()
    total_reward = 0

    while True:
        processed = op({'image': obs, 'action': action})
        with torch.no_grad():
            next_obs_pred, reward_pred = agent.predict(processed)

        # Use predictions for planning/control
        action = agent.plan(obs, next_obs_pred, reward_pred)
        obs, reward, done = env.step(action)
        total_reward += reward

        if done:
            break

    print(f"Episode {episode}: {total_reward}")
```

## Troubleshooting

### Memory Issues
- Use smaller batch sizes
- Enable gradient checkpointing
- Clear cache: `torch.cuda.empty_cache()`

### Speed Issues
- Move to GPU
- Use JIT compilation
- Batch inputs when possible

### Accuracy Issues
- Ensure proper preprocessing with operators
- Check model loading
- Verify input shapes match training