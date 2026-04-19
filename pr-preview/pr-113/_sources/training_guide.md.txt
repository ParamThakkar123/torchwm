# Training Guide

This guide covers how to train world models in TorchWM.

## Overview

TorchWM supports training multiple world model algorithms with a unified interface.

## Basic Training Flow

1. Select an algorithm and create config
2. Set environment/dataset parameters
3. Configure training hyperparameters
4. Initialize agent and call train()

## Dreamer Training

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

# Configure
cfg = DreamerConfig()
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.total_steps = 1_000_000

# Train
agent = DreamerAgent(cfg)
agent.train()
```

## JEPA Training

```python
from world_models.models import JEPAAgent
from world_models.configs import JEPAConfig

cfg = JEPAConfig()
cfg.dataset = "imagenet"
cfg.batch_size = 64
cfg.epochs = 100

agent = JEPAAgent(cfg)
agent.train()
```

## IRIS Training

```python
from world_models.models import IRISAgent
from world_models.configs import IRISConfig

cfg = IRISConfig()
cfg.env_name = "Pong-v5"
cfg.total_epochs = 100

agent = IRISAgent(cfg)
agent.train()
```

## Custom Training Loop

For advanced users, implement custom training:

```python
from world_models.memory import DreamerMemory
from world_models.models import DreamerAgent

agent = DreamerAgent(cfg)
memory = DreamerMemory(cfg)

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
```python
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
```

### Gym
```python
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"
```

### Unity ML-Agents
```python
cfg.env_backend = "unity_mlagents"
cfg.unity_file_name = "path/to/env.exe"
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs
```

### Weights & Biases
```python
cfg.enable_wandb = True
cfg.wandb_project = "torchwm"
cfg.wandb_entity = "your-entity"
```

## Checkpointing

Models are automatically saved:

```python
# Resume training
cfg.restore = True
cfg.checkpoint_path = "path/to/checkpoint"
```

## Distributed Training

For multi-GPU training:

```python
cfg.num_gpus = 4
# TorchWM handles distributed setup automatically
```

## Best Practices

1. **Start small**: Use short episodes and few steps for debugging
2. **Monitor metrics**: Watch loss curves and environment rewards
3. **Tune hyperparameters**: Adjust learning rates and batch sizes
4. **Use checkpoints**: Save frequently and resume from failures
5. **Log experiments**: Use WandB or TensorBoard for tracking