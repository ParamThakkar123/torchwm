# Tutorial: Modular RSSM

The Modular RSSM provides flexible, swappable components for world model research.
Design your own RSSM by mixing and matching encoders, decoders, and backbones.

## Why Modular RSSM?

Traditional RSSM implementations hardcode specific architectures. The modular
design allows researchers to:

- Swap encoders (CNN, ViT, MLP) without changing the rest
- Experiment with different recurrent backbones (GRU, LSTM, Transformer)
- Customize decoder architectures for different observation types
- Rapidly prototype new combinations

## Architecture Overview

```{figure} /_static/modular_rssm.png
:alt: Modular RSSM Architecture
:width: 700px

Modular RSSM with swappable components
```

## Quick Start

### Factory Function

```python
from world_models.models.modular_rssm import create_modular_rssm

rssm = create_modular_rssm(
    encoder_type="conv",       # "conv" | "mlp" | "vit"
    decoder_type="conv",       # "conv" | "mlp"
    backbone_type="gru",       # "gru" | "lstm" | "transformer"
    obs_shape=(3, 64, 64),
    action_size=6,
    stoch_size=32,
    deter_size=200,
    embed_size=1024,
)
```

### Manual Construction

```python
from world_models.models.modular_rssm import ModularRSSM
from world_models.models.modular_rssm import (
    ConvEncoder, ViTEncoder, MLPEncoder,
    ConvDecoder, MLPDecoder,
    GRUBackbone, LSTMBackbone, TransformerBackbone
)

# Custom encoder
encoder = ViTEncoder(
    input_shape=(3, 64, 64),
    embed_size=1024,
    patch_size=8,
    depth=6
)

# Custom backbone
backbone = TransformerBackbone(
    stoch_size=32,
    deter_size=200,
    action_size=6,
    embed_size=1024,
    num_heads=8,
    num_layers=3
)

# Custom decoder
decoder = ConvDecoder(
    stoch_size=32,
    deter_size=200,
    output_shape=(3, 64, 64)
)

# Assemble
rssm = ModularRSSM(
    encoder=encoder,
    decoder=decoder,
    backbone=backbone
)
```

## Component Details

### Encoders

#### ConvEncoder

```python
from world_models.models.modular_rssm import ConvEncoder

encoder = ConvEncoder(
    input_shape=(3, 64, 64),
    embed_size=1024,
    hidden_channels=[32, 64, 128]
)
```

**Architecture:**
```
Input (3, 64, 64)
    → Conv(32, 4x4, s=2) → ReLU
    → Conv(64, 4x4, s=2) → ReLU
    → Conv(128, 4x4, s=2) → ReLU
    → Flatten → Linear(1024)
    → Output (1024,)
```

#### ViTEncoder

```python
from world_models.models.modular_rssm import ViTEncoder

encoder = ViTEncoder(
    input_shape=(3, 64, 64),
    embed_size=1024,
    patch_size=8,
    depth=6,
    num_heads=8
)
```

#### MLPEncoder

```python
from world_models.models.modular_rssm import MLPEncoder

encoder = MLPEncoder(
    input_shape=(3 * 64 * 64,),
    embed_size=1024,
    hidden_sizes=[2048, 1024]
)
```

### Backbones

#### GRUBackbone

```python
from world_models.models.modular_rssm import GRUBackbone

backbone = GRUBackbone(
    stoch_size=32,
    deter_size=200,
    action_size=6,
    embed_size=1024,
    hidden_size=200
)
```

**Mathematics:**
```{math}
h_t = \text{GRU}([s_{t-1}; a_{t-1}], h_{t-1})
```

#### TransformerBackbone

```python
from world_models.models.modular_rssm import TransformerBackbone

backbone = TransformerBackbone(
    stoch_size=32,
    deter_size=200,
    action_size=6,
    embed_size=1024,
    num_heads=8,
    num_layers=3,
    dropout=0.1
)
```

### Decoders

#### ConvDecoder

```python
from world_models.models.modular_rssm import ConvDecoder

decoder = ConvDecoder(
    stoch_size=32,
    deter_size=200,
    output_shape=(3, 64, 64),
    hidden_channels=[128, 64, 32]
)
```

#### MLPDecoder

```python
from world_models.models.modular_rssm import MLPDecoder

decoder = MLPDecoder(
    stoch_size=32,
    deter_size=200,
    output_shape=(3 * 64 * 64,),
    hidden_sizes=[512, 1024, 3 * 64 * 64]
)
```

## Complete Training Example

```python
import torch
import torch.nn as nn
from world_models.models.modular_rssm import create_modular_rssm

# Create modular RSSM
rssm = create_modular_rssm(
    encoder_type="conv",
    decoder_type="conv",
    backbone_type="gru",
    obs_shape=(3, 64, 64),
    action_size=6,
    stoch_size=32,
    deter_size=200,
    embed_size=1024,
)

# Optimizer
optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    # Simulate data
    obs = torch.randn(32, 3, 64, 64)
    action = torch.randn(32, 6)
    next_obs = torch.randn(32, 3, 64, 64)
    reward = torch.randn(32, 1)
    
    # Forward pass
    embed = rssm.encoder(obs)
    hidden = torch.zeros(32, 200)
    stoch = torch.zeros(32, 32)
    
    # Collect sequence
    embeds = [embed]
    stochs = [stoch]
    hiddens = [hidden]
    rewards_pred = []
    obs_pred = []
    
    # Unroll timesteps
    for t in range(16):
        hidden, stoch, prior, posterior = rssm(hidden, stoch, action[:, t], embed)
        
        # Predict
        obs_recon = rssm.decoder(stoch, hidden)
        reward_pred = rssm.reward(stoch, hidden)
        
        embeds.append(embed)
        stochs.append(stoch)
        hiddens.append(hidden)
        rewards_pred.append(reward_pred)
        obs_pred.append(obs_recon)
    
    # Compute loss
    recon_loss = sum(F.mse_loss(o, next_obs) for o in obs_pred) / len(obs_pred)
    reward_loss = F.mse_loss(torch.cat(rewards_pred).squeeze(), reward.squeeze())
    
    # KL loss (simplified)
    kl_loss = sum(
        kl_divergence(p[0], p[1], q[0], q[1]).mean()
        for p, q in zip(prior, posterior)
    ) / len(prior)
    
    loss = recon_loss + 0.1 * kl_loss + reward_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

def kl_divergence(mu1, logstd1, mu2, logstd2):
    var1 = logstd1.exp().pow(2)
    var2 = logstd2.exp().pow(2)
    kl = logstd2 - logstd1 + 0.5 * (var1 + (mu1 - mu2).pow(2)) / var2 - 0.5
    return kl.sum(dim=-1)
```

## Custom Components

### Custom Encoder Example

```python
import torch.nn as nn
from world_models.models.modular_rssm import BaseEncoder

class CustomEncoder(BaseEncoder):
    """Custom encoder example"""
    
    def __init__(self, input_shape, embed_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(256 * 4 * 4, embed_size)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)
```

### Custom Backbone Example

```python
import torch.nn as nn
from world_models.models.modular_rssm import BaseBackbone

class CustomBackbone(BaseBackbone):
    """Custom recurrent backbone"""
    
    def __init__(self, stoch_size, deter_size, action_size, embed_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=stoch_size + action_size,
            hidden_size=deter_size,
            num_layers=2,
            batch_first=True
        )
        self.embed = nn.Linear(deter_size, embed_size)
    
    def forward(self, hidden, stoch, action):
        # Handle single step
        x = torch.cat([stoch, action], dim=-1).unsqueeze(1)
        output, (h_n, c_n) = self.lstm(x, (hidden.unsqueeze(0), 
                                            torch.zeros_like(hidden).unsqueeze(0)))
        return h_n.squeeze(0), output.squeeze(1)
```

## Comparison: Encoders

| Encoder | Best For | Parameters | Speed |
|---------|----------|------------|-------|
| Conv | Images, spatial data | Medium | Fast |
| ViT | Large images, global context | High | Medium |
| MLP | Small inputs, simple features | Low | Fast |

## Comparison: Backbones

| Backbone | Best For | Memory | Captures |
|----------|----------|--------|----------|
| GRU | Standard RL | Low | Short-term |
| LSTM | Long sequences | Medium | Long-term |
| Transformer | Global context | High | All ranges |

## Integration with Dreamer

```python
from world_models.models.modular_rssm import create_modular_rssm
from world_models.configs import DreamerConfig

# Create modular RSSM
rssm = create_modular_rssm(
    encoder_type="vit",       # Use ViT encoder
    decoder_type="conv",
    backbone_type="transformer",
    obs_shape=(3, 64, 64),
    action_size=6,
    stoch_size=32,
    deter_size=200,
    embed_size=1024,
)

# Pass to DreamerConfig
cfg = DreamerConfig()
cfg.rssm = rssm  # Use custom RSSM

agent = DreamerAgent(cfg)
agent.train()
```

## Next Steps

- Read the API Reference for all available components
- Check out advanced training techniques
- Explore diffusion models for world modeling
