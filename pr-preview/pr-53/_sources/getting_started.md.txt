# Getting Started

## Installation

Install from PyPI:

```bash
pip install torchwm
```

Install from source:

```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Logging with Weights & Biases and TensorBoard

TorchWM supports logging experiment results to Weights & Biases (WandB) and TensorBoard.

### Weights & Biases

To use WandB logging, you must provide an API key as anonymous logins are no longer supported.

1. Get your WandB API key from [wandb.ai](https://wandb.ai/settings).
2. Set the key in your config:

```python
cfg.enable_wandb = True
cfg.wandb_api_key = "your-api-key-here"
cfg.wandb_project = "torchwm"
cfg.wandb_entity = "your-entity"
```

### TensorBoard

Enable TensorBoard logging:

```python
cfg.enable_tensorboard = True
cfg.log_dir = "runs"
```

Logs will be saved to the specified directory and can be viewed with `tensorboard --logdir runs`.

## Quick Start: Dreamer

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"
cfg.total_steps = 10_000

agent = DreamerAgent(cfg)
agent.train()
```

## Quick Start: JEPA

```python
from world_models.models import JEPAAgent
from world_models.configs import JEPAConfig

cfg = JEPAConfig()
cfg.dataset = "imagefolder"
cfg.root_path = "./data"
cfg.image_folder = "train"
cfg.epochs = 10

agent = JEPAAgent(cfg)
agent.train()
```

## Environment Backends

Dreamer supports multiple backends through `DreamerConfig.env_backend`:

- `dmc`: DeepMind Control Suite tasks (for example `walker-walk`)
- `gym`: Gym/Gymnasium environment IDs or an existing environment instance
- `unity_mlagents`: Unity ML-Agents executable environments

Important Unity settings are available in `DreamerConfig`:
- `unity_file_name`
- `unity_behavior_name`
- `unity_no_graphics`
- `unity_time_scale`

## Typical Training Flow

1. Create a config object (`DreamerConfig` or `JEPAConfig`).
2. Override dataset/environment and optimization fields.
3. Instantiate the corresponding agent (`DreamerAgent`, `JEPAAgent`).
4. Call `train()` and monitor logs/checkpoints.

For complete API details, see {doc}`api_reference`.
