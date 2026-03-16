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

## Available Algorithms

TorchWM implements multiple world model algorithms. Click on each to see detailed documentation:

| Algorithm | Description | Quick Start |
|-----------|-------------|--------------|
| **Dreamer** | Model-based RL with latent dynamics | {doc}`dreamer` |
| **JEPA** | Self-supervised visual representations | {doc}`jepa` |
| **IRIS** | Sample-efficient RL with Transformers | {doc}`iris` |
| **DiT** | Diffusion models with Transformers | {doc}`dit` |

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

## Typical Training Flow

1. Choose an algorithm (Dreamer, JEPA, IRIS, or DiT)
2. Create a config object for that algorithm
3. Override dataset/environment and optimization fields
4. Instantiate the corresponding agent
5. Call `train()` and monitor logs/checkpoints

For complete API details, see {doc}`api_reference`.
