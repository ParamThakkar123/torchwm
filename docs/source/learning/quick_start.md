# Quick Start Guide

Get up and running with TorchWM in minutes. This guide walks through
installation and your first training run.

## Installation

### From PyPI

```bash
pip install torchwm
```

### From Source

```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Your First Training Run

### Train Dreamer on Gym Environment

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

# Configure the agent
cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"  # Try: CartPole-v1, MountainCarContinuous-v0
cfg.total_steps = 10_000

# Create and train
agent = DreamerAgent(cfg)
agent.train()
```

### Train JEPA on Images

```python
from world_models.models import JEPAAgent
from world_models.configs import JEPAConfig

# Configure JEPA
cfg = JEPAConfig()
cfg.dataset = "imagefolder"
cfg.root_path = "./data/my_images"
cfg.image_folder = "train"
cfg.epochs = 100

# Create and train
agent = JEPAAgent(cfg)
agent.train()
```

## Project Structure

```
torchwm/
├── world_models/
│   ├── models/           # Agents and architectures
│   ├── configs/          # Configuration classes
│   ├── envs/             # Environment adapters
│   ├── datasets/         # Data loaders
│   ├── memory/           # Replay buffers
│   └── utils/            # Utilities
├── docs/                 # Documentation
└── examples/             # Example scripts
```

## Environment Backends

### DeepMind Control Suite

```python
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.env_seed = 0
```

### Gym/Gymnasium

```python
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"
```

### Unity ML-Agents

```python
cfg.env_backend = "unity_mlagents"
cfg.unity_file_name = r"C:\path\to\env.exe"
cfg.unity_behavior_name = "Agent"
cfg.unity_no_graphics = True
```

## Logging

### TensorBoard

```python
cfg.enable_tensorboard = True
cfg.log_dir = "runs/experiment_1"
```

View with:
```bash
tensorboard --logdir runs
```

### Weights & Biases

```python
cfg.enable_wandb = True
cfg.wandb_api_key = "your-api-key"
cfg.wandb_project = "my-project"
cfg.wandb_entity = "your-username"
```

## Common First Steps

### Check Available Environments

```python
from world_models.envs import list_environments

envs = list_environments()
print(envs)
```

### Test Environment Connection

```python
from world_models.envs import make_env
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "CartPole-v1"

env = make_env(cfg)
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")
obs, reward, term, trunc, _ = env.step(env.action_space.sample())
env.close()
```

### Inspect Model Architecture

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig
import torch

cfg = DreamerConfig()
agent = DreamerAgent(cfg)

# Print model summary
print(agent.model)

# Count parameters
total_params = sum(p.numel() for p in agent.model.parameters())
trainable = sum(p.numel() for p in agent.model.parameters() if p.requires_grad)
print(f"Total: {total_params:,}, Trainable: {trainable:,}")
```

## What's Next?

- **Learn the concepts**: Read "Introduction to World Models"
- **Deep dive into Dreamer**: See the Dreamer tutorial
- **Explore JEPA**: Check out self-supervised learning
- **Customize**: Build with Modular RSSM

## Example Projects

### Simple CartPole Agent

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "CartPole-v1"
cfg.total_steps = 50_000

agent = DreamerAgent(cfg)
agent.train()
```

### Continuous Control

```python
cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "HalfCheetah-v4"
cfg.total_steps = 500_000
cfg.batch_size = 64

agent = DreamerAgent(cfg)
agent.train()
```

### Image-based Learning

```python
cfg = DreamerConfig()
cfg.env_backend = "dmc"
cfg.env = "cartpole-balance"
cfg.image_size = 64

agent = DreamerAgent(cfg)
agent.train()
```
