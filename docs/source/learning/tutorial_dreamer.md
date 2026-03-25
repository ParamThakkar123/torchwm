# Tutorial: Dreamer Agent

Dreamer is a model-based reinforcement learning algorithm that learns a world model
and uses it for planning via imagined trajectories. This tutorial covers both the
theory and TorchWM implementation.

## Overview

```{figure} /_static/dreamer_architecture.png
:alt: Dreamer Architecture
:width: 700px

Dreamer combines world model learning with actor-critic optimization
```

## The Dreamer Algorithm

### 1. World Model Learning

The world model learns to predict future latent states and observations:

```{math}
\underbrace{p(s_{t+1} | s_t, a_t)}_{\text{transition}} 
\underbrace{p(o_{t+1} | s_{t+1})}_{\text{observation}}
\underbrace{p(r_t | s_t, a_t)}_{\text{reward}}
```

### 2. Latent Imagination

Once trained, Dreamer can "dream" trajectories without environment interaction:

```{math}
\hat{s}_{t+1} \sim p(s_{t+1} | \hat{s}_t, \hat{a}_t)
\hat{r}_t = p(r_t | \hat{s}_t, \hat{a}_t)
```

### 3. Actor-Critic Training

The actor (policy) and critic (value function) are trained on imagined trajectories:

```{math}
\mathcal{L}_{\text{actor}} = -\mathbb{E}_{\tau \sim p_\theta}[\sum_t \lambda_t \cdot V_\psi(s_t)]
```

## Implementation in TorchWM

### Step 1: Configuration

```python
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"
cfg.total_steps = 100_000
cfg.batch_size = 50
cfg.chunk_size = 64
cfg.lr = 1e-4
cfg.gamma = 0.99
cfg.lambda_ = 0.95  # TD(lambda) decay
```

### Step 2: Create the Agent

```python
from world_models.models import DreamerAgent

agent = DreamerAgent(cfg)
```

### Step 3: Train

```python
agent.train()
```

## Deep Dive: The Mathematics

### RSSM State Evolution

The RSSM maintains two types of state:

1. **Deterministic hidden state** :math:`h_t`: Captures temporal dependencies via RNN
2. **Stochastic latent state** :math:`s_t`: Discretized bottleneck for information

```{math}
h_t = \text{GRU}(h_{t-1}, [s_{t-1}; a_{t-1}])

s_t \sim q(s_t | h_t, o_t) \quad \text{(posterior)}

\hat{s}_t \sim p(s_t | h_{t-1}, a_{t-1}) \quad \text{(prior)}
```

### KL Balancing

To prevent posterior collapse, we use KL balancing:

```python
def kl_loss(prior_mean, prior_logstd, posterior_mean, posterior_logstd, 
            kl_balance=0.8):
    """KL balancing - use mix of targets to prevent posterior collapse"""
    # Target: 0.8 * posterior_target + 0.2 * prior_target
    sg = lambda x: x.detach()  # stop gradient
    
    kl_forward = kl_divergence(
        posterior_mean, posterior_logstd,
        sg(prior_mean), sg(prior_logstd)
    )
    kl_reverse = kl_divergence(
        sg(prior_mean), sg(prior_logstd),
        posterior_mean, posterior_logstd
    )
    
    return kl_balance * kl_forward + (1 - kl_balance) * kl_reverse

def kl_divergence(mu1, logstd1, mu2, logstd2):
    """KL divergence between two Gaussians"""
    var1 = logstd1.exp().pow(2)
    var2 = logstd2.exp().pow(2)
    
    kl = logstd2 - logstd1 + 0.5 * (var1 + (mu1 - mu2).pow(2)) / var2 - 0.5
    return kl.sum(dim=-1)
```

### TD(λ) Returns

Dreamer uses λ-returns for credit assignment:

```python
def compute_lambda_returns(rewards, values, lambda_=0.95, gamma=0.99):
    """Compute TD(λ) returns
    
    Where G_t^{(n)} is the n-step return
    """
    T = rewards.size(0)
    returns = torch.zeros_like(rewards)
    
    # Bootstrap from the end
    returns[-1] = rewards[-1] + gamma * values[-1]
    
    # Propagate backwards
    for t in reversed(range(T - 1)):
        returns[t] = rewards[t] + gamma * (
            (1 - lambda_) * values[t + 1] + lambda_ * returns[t + 1]
        )
    
    return returns

def compute_value_loss(pred_values, target_values, loss_scale=1.0):
    """Value function loss"""
    return loss_scale * F.mse_loss(pred_values, target_values.detach())
```

### Actor (Policy) Loss

The actor maximizes expected imagined returns:

```python
def actor_loss(model, imagined_trajectories, lambda_=0.95, gamma=0.99):
    """Actor loss - maximize expected returns
    
    Loss = -E[Σ_t γ^t * λ_t * V(s_t)]
    """
    T, B = imagined_trajectories['rewards'].shape
    
    # Compute returns for each timestep
    discounts = (gamma ** torch.arange(T, device=rewards.device)).unsqueeze(1)
    
    # Value function prediction
    values = model.value(imagined_trajectories['stoch'], 
                         imagined_trajectories['hidden'])
    
    # λ-returns
    lambda_returns = compute_lambda_returns(
        rewards, values, lambda_, gamma
    )
    
    # Weighted loss
    weighted_returns = (discounts * lambda_returns).sum() / B
    
    return -weighted_returns
```

## Customizing Dreamer

### Using Different Environment Backends

```python
# DeepMind Control Suite
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.env_seed = 42

# Unity ML-Agents
cfg.env_backend = "unity_mlagents"
cfg.unity_file_name = r"C:\path\to\env.exe"
cfg.unity_behavior_name = "Agent"
cfg.unity_no_graphics = True
cfg.unity_time_scale = 20.0
```

### Custom Encoder/Decoder

```python
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.encoder_type = "vit"      # Vision Transformer encoder
cfg.decoder_type = "mlp"      # MLP decoder (for vector observations)
cfg.backbone_type = "lstm"    # LSTM instead of GRU
```

### Using Modular RSSM

```python
from world_models.models.modular_rssm import create_modular_rssm

# Fully modular RSSM
rssm = create_modular_rssm(
    encoder_type="vit",         # Multiple encoder choices
    decoder_type="conv",
    backbone_type="transformer", # Transformer backbone
    obs_shape=(3, 64, 64),
    action_size=6,
    stoch_size=32,
    deter_size=200,
    embed_size=1024,
)
```

## Complete Training Example

```python
import torch
from world_models.configs import DreamerConfig
from world_models.models import DreamerAgent
from world_models.envs import make_env

# 1. Configuration
cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "CartPole-v1"
cfg.total_steps = 50_000
cfg.batch_size = 50
cfg.chunk_size = 16

# 2. Create agent
agent = DreamerAgent(cfg)

# 3. Training loop (manual control)
env = make_env(cfg)

state, _ = env.reset()
episode_reward = 0
episode_count = 0

while agent.global_step < cfg.total_steps:
    # Sample from replay buffer
    if len(agent.replay) >= cfg.batch_size:
        batch = agent.replay.sample(cfg.batch_size, cfg.chunk_size)
        
        # World model training
        model_loss = agent.train_world_model(batch)
        
        # Actor-Critic training (every few steps)
        if agent.global_step % cfg.actor_update_freq == 0:
            actor_loss, critic_loss = agent.train_actor_critic()
    
    # Environment interaction
    action = agent.get_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    agent.replay.add(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward
    
    if done:
        print(f"Episode {episode_count}: {episode_reward}")
        state, _ = env.reset()
        episode_reward = 0
        episode_count += 1

print("Training complete!")
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/
```

### Weights & Biases

```python
cfg.enable_wandb = True
cfg.wandb_api_key = "your-api-key"
cfg.wandb_project = "my-dreamer-experiments"
cfg.wandb_entity = "your-username"
```

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Posterior collapse | Increase `kl_free`, reduce `kl_scale` |
| Poor exploration | Increase `action_noise` in config |
| Slow training | Reduce `chunk_size`, increase `batch_size` |
| Unstable learning | Reduce `lr`, increase `target_update_freq` |

## Next Steps

- Learn about JEPA for self-supervised representation learning
- Explore the Modular RSSM for research experiments
- Check the API Reference for detailed documentation
