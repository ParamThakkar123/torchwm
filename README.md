# TorchWM

A modular PyTorch library for learning, training, and deploying world models across various environments. This package provides minimal implementations of popular world model algorithms, enabling researchers and developers to experiment with predictive modeling in reinforcement learning and beyond.

## Features

- **Modular Design**: Easily extensible components for encoders, decoders, transition models, and reward predictors.
- **Supported Algorithms**:
  - Dreamer (v1 and v2 variants)
  - PlaNet
  - World Model-based agents for custom environments
- **Integration**: Compatible with DMC, MuJoCo, Atari, Gym/Gymnasium, and Unity ML-Agents environments.
- **Evaluation Tools**: Built-in scripts for training, evaluation, and visualization.
- **PyTorch Native**: Leverages PyTorch's dynamic computation graphs for efficient training.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- MuJoCo (for physics simulations)

### Install from PyPI
```bash
pip install torchwm
```

### Install from Source
Clone the repository and install in editable mode:
```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm
pip install -e .
```

For development dependencies (testing, linting):
```bash
pip install -e ".[dev]"
```

## Quick Start

### Training a Dreamer Agent
```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"   # "dmc", "gym", or "unity_mlagents"
cfg.env = "Pendulum-v1"
cfg.total_steps = 10000

agent = DreamerAgent(cfg)
agent.train()
```

### Training Dreamer on Unity ML-Agents
```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "unity_mlagents"
cfg.unity_file_name = r"E:\UnityBuilds\MyEnv.exe"
cfg.unity_behavior_name = "MyBehavior"
cfg.unity_no_graphics = True
cfg.unity_time_scale = 20.0

agent = DreamerAgent(cfg)
agent.train()
```

### Training PlaNet/RSSM with Unity ML-Agents
```python
from world_models.models import Planet
from world_models.envs import UnityMLAgentsEnv

unity_env = UnityMLAgentsEnv(
    file_name=r"E:\UnityBuilds\MyEnv.exe",
    behavior_name="MyBehavior",
    no_graphics=True,
)

planet = Planet(env=unity_env, bit_depth=5)
planet.train(epochs=10)
```

### Evaluating a Trained Model
Use the provided evaluation script:
```bash
python dreamer_eval.py --model_path results/dreamer_v1_custom_env/model.pth --env Pendulum-v1
```

### Custom Environment Example
```python
from world_models import WorldModel
import torch

# Define your custom environment
class CustomEnv:
    def __init__(self):
        self.obs_dim = 10
        self.act_dim = 2

env = CustomEnv()
model = WorldModel(obs_dim=env.obs_dim, act_dim=env.act_dim)
# Train or load model...
```

## Project Structure

- `world_models/`: Core library modules (encoders, decoders, agents)
- `dreamer_eval.py`: Evaluation script for Dreamer agents
- `dreamer_try.py`: Quick try-out script for Dreamer
- `main.py`: Main entry point for training
- `results/`: Directory for storing trained models and logs

## Documentation

For detailed API documentation, see the [Wiki](https://github.com/ParamThakkar123/torchwm/wiki) or docstrings in the source code.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Key areas:
- Adding new world model algorithms
- Improving existing implementations
- Bug fixes and performance optimizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library for your research, please cite:

```bibtex
@misc{Thakkar_GitHub_-_ParamThakkar123_torchwm,
author = {Thakkar, Param},
title = {{GitHub - ParamThakkar123/torchwm: A modular PyTorch library designed for learning, training, and deploying world models across various environments.}},
year = {2025},
url = {https://github.com/ParamThakkar123/torchwm}
}
```

Package link: [TorchWM](https://pypi.org/project/torchwm/)
