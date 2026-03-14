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

## Quick Start: Modular RSSM

The modular RSSM allows researchers to swap encoders, decoders, and backbones for experimentation:

```python
from world_models.models.modular_rssm import create_modular_rssm, ModularRSSM
from world_models.models.modular_rssm import ConvEncoder, ViTEncoder, GRUBackbone, LSTMBackbone

# Factory function for quick setup
rssm = create_modular_rssm(
    encoder_type="conv",      # "conv", "mlp", or "vit"
    decoder_type="conv",       # "conv" or "mlp"
    backbone_type="gru",      # "gru", "lstm", or "transformer"
    obs_shape=(3, 64, 64),
    action_size=6,
    stoch_size=32,
    deter_size=200,
    embed_size=1024,
)

# Or build manually with custom components
encoder = ViTEncoder(input_shape=(3, 64, 64), embed_size=1024, patch_size=8, depth=6)
backbone = LSTMBackbone(action_size=6, stoch_size=32, deter_size=200, hidden_size=200, embed_size=1024)
rssm = ModularRSSM(encoder=encoder, decoder=decoder, backbone=backbone, reward_decoder=reward_decoder)
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
