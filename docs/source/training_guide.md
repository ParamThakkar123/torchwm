# Training Guide

This guide covers how to train world models in TorchWM.

## Overview

TorchWM supports training multiple world model algorithms with a unified interface.

## Basic Training Flow

1. Select an algorithm
2. Override environment, dataset, or optimization parameters
3. Initialize the agent
4. Call `train()` and monitor logs/checkpoints

The simplest path is the top-level `torchwm` API:

```python :class: thebe
import torchwm

agent = torchwm.create_model(
    "dreamer",
    env_backend="dmc",
    env="walker-walk",
    total_steps=1_000_000,
)
agent.train()
```

For research code, the lower-level config and agent classes remain available.

## Dreamer Training

Preferred application API:

```python :class: thebe
import torchwm

agent = torchwm.create_model(
    "dreamer",
    env_backend="dmc",
    env="walker-walk",
    total_steps=1_000_000,
)
agent.train()
```

Equivalent direct API:

```python :class: thebe
from torchwm import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.total_steps = 1_000_000

agent = DreamerAgent(cfg)
agent.train()
```

## JEPA Training

```python :class: thebe
import torchwm

agent = torchwm.create_model(
    "jepa",
    dataset="imagenet",
    batch_size=64,
    epochs=100,
)
agent.train()
```

## IRIS Training

`IRISAgent` needs constructor arguments such as `action_size` and `device` in
addition to its config, so pass those as constructor overrides:

```python :class: thebe
import torch
import torchwm

agent = torchwm.create_model(
    "iris",
    env_name="Pong-v5",
    total_epochs=100,
    action_size=4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```

## Custom Training Loop

For advanced users, implement custom training:

```python :class: thebe
from torchwm import DreamerAgent, ReplayBuffer

agent = DreamerAgent(cfg)
memory = ReplayBuffer(
    size=100_000,
    obs_shape=(3, 64, 64),
    action_size=6,
    seq_len=50,
    batch_size=50,
)

for step in range(cfg.total_steps):
    # Collect experience
    experience = agent.collect_episode()
    memory.add_episode(experience)

    # Train
    if step % cfg.update_steps == 0:
        batch = memory.sample_batch()
        metrics = agent.update(batch)

    # Log
    if step % cfg.log_every == 0:
        print(f"Step {step}: {metrics}")
```

## Configuration

All training is controlled via config objects:

### Common Parameters
- `seed`: Random seed
- `device`: Training device
- `total_steps`/`epochs`: Training duration
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `grad_clip_norm`: Gradient clipping

### Logging
- `enable_wandb`: Weights & Biases logging
- `log_dir`: TensorBoard log directory
- `checkpoint_interval`: Save frequency

## Environment Setup

### DMC
```python :class: thebe
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
```

### DeepMind Lab
```python :class: thebe
cfg.env_backend = "dmlab"
cfg.env = "rooms_collect_good_objects_train"
cfg.dmlab_action_repeat = 4
```

### Gym
```python :class: thebe
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"
```

### Brax
```python :class: thebe
cfg.env_backend = "brax"
cfg.env = "ant"
cfg.brax_backend = "generalized"
```

### Unity ML-Agents
```python :class: thebe
cfg.env_backend = "unity_mlagents"
cfg.unity_file_name = "path/to/env.exe"
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs
```

### Weights & Biases
```python :class: thebe
cfg.enable_wandb = True
cfg.wandb_project = "torchwm"
cfg.wandb_entity = "your-entity"
```

## Checkpointing

Models are automatically saved:

```python :class: thebe
# Resume training
cfg.restore = True
cfg.checkpoint_path = "path/to/checkpoint"
```

## Distributed Training

For multi-GPU training:

```python :class: thebe
cfg.num_gpus = 4
# TorchWM handles distributed setup automatically
```

## Best Practices

1. **Start small**: Use short episodes and few steps for debugging
2. **Monitor metrics**: Watch loss curves and environment rewards
3. **Tune hyperparameters**: Adjust learning rates and batch sizes
4. **Use checkpoints**: Save frequently and resume from failures
5. **Log experiments**: Use WandB or TensorBoard for tracking