# Introduction to World Models

World models enable agents to learn a compact representation of their environment
and simulate how it evolves over time. This foundational understanding will help
you grasp why TorchWM is structured the way it is.

## What is a World Model?

A world model learns to predict future observations (and rewards) given past
observations and actions. Instead of relying solely on real environment
interactions, the agent can "dream" about potential futures.

```{math}
p(s_{t+1} | s_t, a_t) \quad \text{and} \quad p(o_{t+1} | s_{t+1})
```

Where:
- :math:`s_t` = latent state at time t
- :math:`a_t` = action taken at time t
- :math:`o_{t+1}` = observation at time t+1

## Why Learn World Models?

1. **Sample Efficiency**: Learn in simulation before real-world interaction
2. **Planning**: Imagine trajectories without executing them
3. **Representation Learning**: Extract meaningful features from raw observations
4. **Temporal Abstraction**: Model long-term dependencies

## The Latent Dynamics Approach

Instead of predicting raw high-dimensional observations, we learn a compressed
latent representation:

```{figure} /_static/world_model_architecture.png
:alt: World Model Architecture
:width: 600px

High-level overview of a latent dynamics model
```

## Core Components

### 1. Encoder

Maps observations to latent embeddings:

```python
# Conceptual encoder
import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, obs_shape, embed_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, embed_size)
    
    def forward(self, obs):
        x = self.features(obs / 255.0)
        return self.fc(x.flatten(1))
```

### 2. Latent Dynamics Model (RSSM)

The Recurrent State-Space Model (RSSM) is the heart of Dreamer:

```python
"""
RSSM Mathematical Formulation:
------------------------------
1. Posterior (encoder output):
   p(s_t | h_{t-1}, a_{t-1}, o_t)
   
2. Prior (predicted):
   p(s_t | h_{t-1}, a_{t-1})
   
3. Recurrent transition:
   h_t = f(h_{t-1}, s_t, a_t)

Where:
- h_t: deterministic hidden state
- s_t: stochastic latent state
"""

class RSSM(nn.Module):
    def __init__(self, stoch_size, deter_size, action_size, embed_size):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        
        # Core transition: h_{t-1}, s_{t-1}, a_{t-1} -> h_t
        self.recurrent = nn.GRUCell(
            input_size=stoch_size + action_size,
            hidden_size=deter_size
        )
        
        # Prior: predict next state distribution
        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, 256),
            nn.ReLU(),
            nn.Linear(256, stoch_size * 2)  # mean + logstd
        )
        
        # Posterior: update with observation
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_size + embed_size, 256),
            nn.ReLU(),
            nn.Linear(256, stoch_size * 2)
        )
    
    def forward(self, prev_hidden, prev_stoch, action, embed):
        # Recurrent core
        hidden = self.recurrent(
            torch.cat([prev_stoch, action], -1),
            prev_hidden
        )
        
        # Compute prior and posterior
        prior_mean, prior_logstd = torch.chunk(
            self.prior_net(hidden), 2, dim=-1
        )
        posterior_mean, posterior_logstd = torch.chunk(
            self.posterior_net(torch.cat([hidden, embed], -1)), 2, dim=-1
        )
        
        # Reparameterization
        stoch = self._reparameterize(posterior_mean, posterior_logstd)
        
        return hidden, stoch, (prior_mean, prior_logstd), (posterior_mean, posterior_logstd)
    
    def _reparameterize(self, mean, logstd):
        std = logstd.exp()
        return mean + std * torch.randn_like(std)
```

### 3. Decoder

Reconstruct observations from latent states:

```python
class ConvDecoder(nn.Module):
    def __init__(self, stoch_size, deter_size, obs_shape):
        super().__init__()
        self.fc = nn.Linear(stoch_size + deter_size, 1024)
        self.reshape = (128, 4, 4)
        
        self.features = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_shape[0], 4, stride=2),
        )
    
    def forward(self, stoch, hidden):
        x = self.fc(torch.cat([stoch, hidden], -1))
        x = x.view(-1, *self.reshape)
        return self.features(x)
```

### 4. Reward/Value Heads

Predict rewards and values for planning:

```python
class RewardHead(nn.Module):
    def __init__(self, stoch_size, deter_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stoch_size + deter_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # scalar reward
        )
    
    def forward(self, stoch, hidden):
        return self.net(torch.cat([stoch, hidden], -1))
```

## Loss Functions

The overall world model loss combines multiple components:

```{math}
\mathcal{L}_{world} = \mathcal{L}_{reconstruction} + \mathcal{L}_{KL} + \mathcal{L}_{reward}
```

### Reconstruction Loss

```python
def reconstruction_loss(pred_obs, target_obs):
    """Reconstruction loss: predicted vs actual observations"""
    # Use MSE or BCE
    return F.mse_loss(pred_obs, target_obs)
```

### KL Divergence

```python
def kl_loss(prior_mean, prior_logstd, posterior_mean, posterior_logstd):
    """KL divergence between prior and posterior
    
    KL(q||p) = 0.5 * (logstd_p - logstd_q + (std_q² + (mu_q - mu_p)²) / std_p² - 1)
    """
    prior_var = prior_logstd.exp().pow(2)
    posterior_var = posterior_logstd.exp().pow(2)
    
    kl = 0.5 * (
        prior_logstd - posterior_logstd 
        + (posterior_var + (posterior_mean - prior_mean).pow(2)) / prior_var 
        - 1
    )
    return kl.mean()
```

## Putting It All Together

Here's how the components connect in TorchWM:

```python
from world_models.models.modular_rssm import create_modular_rssm
from world_models.vision.dreamer_encoder import ConvEncoder
from world_models.vision.dreamer_decoder import ConvDecoder
from world_models.reward import RewardHead

# Create complete world model
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

# Forward pass
hidden = torch.zeros(1, 200)  # deter_size
stoch = torch.zeros(1, 32)     # stoch_size
action = torch.randn(1, 6)     # action_size
obs = torch.randn(1, 3, 64, 64)  # observation

# 1. Encode observation
embed = rssm.encoder(obs)

# 2. RSSM transition
hidden, stoch, prior, posterior = rssm(
    hidden, stoch, action, embed
)

# 3. Decode/reconstruct
reconstructed_obs = rssm.decoder(stoch, hidden)

# 4. Predict reward
reward = rssm.reward(stoch, hidden)

print(f"Latent state shape: {stoch.shape}")
print(f"Reconstructed obs shape: {reconstructed_obs.shape}")
print(f"Predicted reward: {reward.item():.4f}")
```

## Next Steps

Now that you understand the core concepts, proceed to the Dreamer tutorial
to see how these components work together in a complete agent:
