# RSSM Deep Dive

This tutorial provides an in-depth mathematical and implementation understanding
of the Recurrent State-Space Model (RSSM), the core component of Dreamer.

## Core Concepts

### The State-Space Model

An RSSM models the world as a latent variable model:

```{math}
p(s_{1:T}, o_{1:T}, a_{1:T}) = p(s_1) \prod_{t=1}^T p(s_t | s_{t-1}, a_{t-1}) p(o_t | s_t) p(a_t | s_{\le t})
```

Where:
- :math:`s_t`: latent state at time t
- :math:`o_t`: observation at time t  
- :math:`a_t`: action at time t

### Two-Part State Representation

RSSM uses a hybrid state representation:

1. **Deterministic state** :math:`h_t`: Captures all deterministic dynamics via RNN
2. **Stochastic state** :math:`s_t`: Stochastic latent variable (discrete or continuous)

## The Mathematical Framework

### State Transition

The latent dynamics are modeled as:

```{math}
h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
```

Where f is typically a GRU or LSTM:

```python
# GRU transition
h_t = \text{GRU}([s_{t-1}; a_{t-1}], h_{t-1})
```

### Prior and Posterior

**Prior** (predicted next state):
```{math}
\hat{s}_t \sim p(s_t | h_{t-1}, a_{t-1}) = \mathcal{N}(\mu_\text{prior}(h_{t-1}), \sigma_\text{prior}(h_{t-1}))
```

**Posterior** (updated with observation):
```{math}
s_t \sim q(s_t | h_t, o_t) = \mathcal{N}(\mu_\text{post}(h_t, o_t), \sigma_\text{post}(h_t, o_t))
```

### Complete RSSM Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RSSM(nn.Module):
    """Complete RSSM implementation
    
    Core components:
    1. GRU: deterministic state transition
    2. Prior Net: predict next state prior distribution
    3. Posterior Net: update posterior with observation
    """
    
    def __init__(
        self,
        stoch_size: int = 32,
        deter_size: int = 200,
        action_size: int = 6,
        embed_size: int = 1024,
        hidden_size: int = 200,
        discrete: bool = False,
        num_discrete: int = 32,
    ):
        super().__init__()
        
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.action_size = action_size
        self.embed_size = embed_size
        self.discrete = discrete
        self.num_discrete = num_discrete if discrete else 0
        
        # GRU for deterministic state
        self.gru = nn.GRUCell(
            input_size=stoch_size + action_size,
            hidden_size=deter_size
        )
        
        # Prior network: h -> prior distribution
        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        
        if discrete:
            self.prior_mean = nn.Linear(hidden_size, num_discrete * stoch_size)
            self.prior_logit = nn.Linear(hidden_size, num_discrete * stoch_size)
        else:
            self.prior_mean = nn.Linear(hidden_size, stoch_size)
            self.prior_logstd = nn.Linear(hidden_size, stoch_size)
        
        # Posterior network: h + embed -> posterior distribution
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_size + embed_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        
        if discrete:
            self.posterior_mean = nn.Linear(hidden_size, num_discrete * stoch_size)
            self.posterior_logit = nn.Linear(hidden_size, num_discrete * stoch_size)
        else:
            self.posterior_mean = nn.Linear(hidden_size, stoch_size)
            self.posterior_logstd = nn.Linear(hidden_size, stoch_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m)
    
    def init_state(self, batch_size: int, device: torch.device):
        """Initialize hidden states"""
        return {
            'hidden': torch.zeros(batch_size, self.deter_size, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_size, device=device),
        }
    
    def forward(self, prev_state, action, embed, training=True):
        """Single step forward pass
        
        Args:
            prev_state: dict with 'hidden' and 'stoch'
            action: (B, action_size)
            embed: (B, embed_size) - encoded observation
            training: bool - whether to compute posterior
        
        Returns:
            new_state: updated hidden and stochastic states
            prior: prior distribution parameters
            posterior: posterior distribution parameters (if training)
        """
        hidden = prev_state['hidden']
        stoch = prev_state['stoch']
        
        # 1. GRU transition
        gru_input = torch.cat([stoch, action], dim=-1)
        hidden = self.gru(gru_input, hidden)
        
        # 2. Prior (predicted next state)
        prior_feat = self.prior_net(hidden)
        
        if self.discrete:
            prior_logits = self.prior_logit(prior_feat).view(
                -1, self.num_discrete, self.stoch_size
            )
            prior_mean = None
            if training:
                stoch = self._sample_discrete(prior_logits)
            else:
                stoch = torch.argmax(prior_logits, dim=1)
        else:
            prior_mean = self.prior_mean(prior_feat)
            prior_logstd = self.prior_logstd(prior_feat)
            prior_logstd = torch.clamp(prior_logstd, -20, 2)
            
            if training:
                stoch = self._reparameterize(prior_mean, prior_logstd)
            else:
                stoch = prior_mean
        
        # 3. Posterior (updated with observation)
        if training and embed is not None:
            posterior_feat = self.posterior_net(torch.cat([hidden, embed], dim=-1))
            
            if self.discrete:
                posterior_logits = self.posterior_logit(posterior_feat).view(
                    -1, self.num_discrete, self.stoch_size
                )
                stoch = self._sample_discrete(posterior_logits)
                posterior = {'logits': posterior_logits}
            else:
                posterior_mean = self.posterior_mean(posterior_feat)
                posterior_logstd = self.posterior_logstd(posterior_feat)
                posterior_logstd = torch.clamp(posterior_logstd, -20, 2)
                stoch = self._reparameterize(posterior_mean, posterior_logstd)
                posterior = {'mean': posterior_mean, 'logstd': posterior_logstd}
        else:
            posterior = None
        
        new_state = {'hidden': hidden, 'stoch': stoch}
        
        prior_info = {'mean': prior_mean, 'logstd': prior_logstd} if not self.discrete else {'logits': prior_logits}
        
        return new_state, prior_info, posterior
    
    def _reparameterize(self, mean, logstd):
        """Reparameterization trick: z = mu + sigma * eps"""
        std = logstd.exp()
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def _sample_discrete(self, logits):
        """Gumbel-Softmax sampling for discrete latent"""
        return F.gumbel_softmax(logits, dim=1, hard=False)
```

## KL Divergence and Free Bits

### The KL Loss

```python
def kl_loss(prior, posterior, kl_free=1.0, kl_scale=1.0, kl_balance=0.8):
    """Compute KL divergence loss
    
    Uses kl_balance to balance forward and backward KL:
    - Forward KL: q(s_t | h_t, o_t) || p(s_t | h_{t-1}, a_{t-1})
    - Backward KL: p(s_t | h_{t-1}, a_{t-1}) || q(s_t | h_t, o_t)
    """
    if 'logits' in prior:  # discrete
        kl = F.kl_div(
            F.log_softmax(prior['logits'], dim=-1),
            F.softmax(posterior['logits'], dim=-1),
            reduction='none'
        ).sum(dim=-1)
    else:  # continuous
        var_posterior = posterior['logstd'].exp().pow(2)
        var_prior = prior['logstd'].exp().pow(2)
        
        kl = 0.5 * (
            prior['logstd'] - posterior['logstd']
            + (var_posterior + (posterior['mean'] - prior['mean']).pow(2)) / var_prior
            - 1
        )
    
    # Free bits: don't update if KL too small
    kl = torch.clamp(kl, min=kl_free)
    
    return kl_scale * kl.mean()
```

### Why KL Balancing?

KL balancing prevents posterior collapse:

```python
# Original: use posterior target only
kl_naive = kl(posterior, prior)

# Balanced: mix prior and posterior targets
kl_balanced = 0.8 * kl(posterior, stopgrad(prior)) + 0.2 * kl(stopgrad(posterior), prior)
```

## Rolling Out Trajectories

### Imagined Rollout

```python
def imagine_rollout(rssm, initial_state, policy, horizon=15):
    """Rollout imagined trajectories in latent space
    
    Args:
        rssm: RSSM model
        initial_state: initial hidden/stoch states
        policy: policy network
        horizon: number of steps to imagine
    
    Returns:
        trajectory: dict with states, actions, rewards
    """
    trajectory = {
        'stoch': [],
        'hidden': [],
        'actions': [],
        'rewards': [],
    }
    
    state = initial_state
    
    for t in range(horizon):
        # Sample action
        action = policy(state['stoch'], state['hidden'])
        
        # RSSM transition (without posterior)
        state, prior, _ = rssm(state, action, None, training=False)
        
        # Predict reward
        reward = rssm.reward_net(state['stoch'], state['hidden'])
        
        trajectory['stoch'].append(state['stoch'])
        trajectory['hidden'].append(state['hidden'])
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
    
    return {k: torch.stack(v, dim=1) for k, v in trajectory.items()}
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from world_models.models.modular_rssm import create_modular_rssm

def train_rssm():
    """Complete RSSM training loop"""
    
    # 1. Create model
    rssm = create_modular_rssm(
        encoder_type="conv",
        decoder_type="conv",
        backbone_type="gru",
        obs_shape=(3, 64, 64),
        action_size=6,
        stoch_size=32,
        deter_size=200,
        embed_size=1024,
    ).cuda()
    
    optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-4)
    
    # 2. Training loop
    for epoch in range(100):
        # Sample batch
        obs = torch.randn(32, 16, 3, 64, 64).cuda()  # (T, B, C, H, W)
        actions = torch.randn(32, 16, 6).cuda()
        rewards = torch.randn(32, 16, 1).cuda()
        
        total_loss = 0
        
        # Initialize state
        state = rssm.init_state(32, obs.device)
        
        for t in range(15):  # chunk size
            # Encode observation
            embed = rssm.encoder(obs[t])
            
            # RSSM forward
            state, prior, posterior = rssm(state, actions[t], embed, training=True)
            
            # Compute loss
            # 1. Reconstruction loss
            obs_pred = rssm.decoder(state['stoch'], state['hidden'])
            recon_loss = F.mse_loss(obs_pred, obs[t])
            
            # 2. KL loss
            kl = kl_loss(prior, posterior, kl_balance=0.8)
            
            # 3. Reward loss
            reward_pred = rssm.reward(state['stoch'], state['hidden'])
            reward_loss = F.mse_loss(reward_pred, rewards[t])
            
            loss = recon_loss + 0.1 * kl + reward_loss
            total_loss += loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: Loss={total_loss.item():.4f}")

def kl_loss(prior, posterior, kl_balance=0.8):
    """KL loss with balancing"""
    if 'logits' in prior:
        # Discrete case
        return F.kl_div(
            F.log_softmax(prior['logits'].view(-1, 32), dim=-1),
            F.softmax(posterior['logits'].view(-1, 32), dim=-1),
            reduction='batchmean'
        )
    else:
        # Continuous case
        var_p = prior['logstd'].exp().pow(2)
        var_q = posterior['logstd'].exp().pow(2)
        
        kl = 0.5 * (
            prior['logstd'] - posterior['logstd']
            + (var_q + (posterior['mean'] - prior['mean']).pow(2)) / var_p
            - 1
        )
        return kl.mean()

if __name__ == "__main__":
    train_rssm()
```

## Advanced Topics

### Discrete vs Continuous Latents

| Aspect | Discrete | Continuous |
|--------|----------|------------|
| Representation | Categorical | Gaussian |
| Posterior collapse | More prone | Less prone |
| Expressiveness | Good for categorical | Good for continuous |
| Implementation | Gumbel-softmax | Reparameterization |

### Hierarchical RSSM

```python
class HierarchicalRSSM(nn.Module):
    """Hierarchical RSSM: multiple time-scale states"""
    
    def __init__(self, stoch_size, deter_size):
        super().__init__()
        # Slow dynamics (high-level)
        self.slow_rssm = RSSM(stoch_size, deter_size)
        # Fast dynamics (low-level)  
        self.fast_rssm = RSSM(stoch_size, deter_size // 4)
```

## References

- [Dreamer: Learning Latent Dynamics for Planning](https://arxiv.org/abs/1912.01603)
- [PlaNet: Learning Latent Dynamics](https://arxiv.org/abs/1811.04551)
