# Dreamer: Model-Based RL with Latent Dynamics

Dreamer is a model-based reinforcement learning algorithm that learns a latent dynamics model
from images and trains a behavior policy entirely in the latent space.

Based on papers:
- [Dreamer: Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1912.01603) (Hafner et al., 2019)
- [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193) (DreamerV2, Hafner et al., 2020)

## Key Idea

Dreamer learns:
1. **World Model**: Latent dynamics model that predicts future latent states
2. **Value Model**: Estimates expected returns from any latent state
3. **Policy**: Actions that maximize expected returns in latent space

The key innovation is learning behaviors purely in imagination - no gradients flow from the environment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         World Model (RSSM)                          │
│                                                                      │
│  ┌─────────┐    ┌───────────────────┐    ┌────────────────────────┐ │
│  │Encoder  │    │    Latent Model   │    │       Decoder          │ │
│  │ (CNN)   │ -> │  (GRU + Stoch)    │ -> │    (Transposed CNN)    │ │
│  │ 64x64   │    │ h_t = f(h_{t-1},  │    │                        │ │
│  │         │    │        s_{t-1}, a)│    │  p(x_t | s_t, h_t)    │ │
│  └─────────┘    └───────────────────┘    └────────────────────────┘ │
│                                                                      │
│        s_t ~ p(s_t | h_t)       h_t ~ p(h_t | s_{t-1}, a_{t-1})    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Imagination Rollout                         │
│                                                                      │
│  s_0 ──► a_0 ──► s_1 ──► a_1 ──► s_2 ──► ... ──► s_H               │
│  │        │                                                            │
│  └────────┴──────────────────────────────────────┐                  │
│                                               ▼                     │
│                      ┌─────────────────────────────┐                │
│                      │      λ-return target        │                │
│                      │  G_t = r_t + γ(1-λ)v + λG_{t+1}               │
│                      └─────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Actor-Critic Learning                        │
│                                                                      │
│  Actor: π(a_t | s_t, h_t)  ──►  REINFORCE with baseline             │
│  Critic: v(s_t, h_t)       ──►  MSE on λ-returns                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Recurrent State Space Model (RSSM)

The core world model combining:
- **Deterministic hidden state** (h_t): Recurrent state (GRU)
- **Stochastic latent state** (s_t): Discrete or continuous latent variables

**Dynamics**: `h_t = f(h_{t-1}, s_{t-1}, a_{t-1})`
**Posterior**: `s_t ~ q(s_t | h_t, x_t)`
**Prior**: `s_t ~ p(s_t | h_t)`

### 2. Encoder/Decoder

- **Encoder**: CNN that maps images to latent embeddings
- **Decoder**: Transposed CNN that reconstructs images from latents
- Both use ReLU activations and residual connections

### 3. Reward/Discount Heads

- **Reward model**: Predicts reward from latent state
- **Discount model**: Predicts episode termination (DreamerV2)

## Training

```python
from world_models.models import DreamerAgent
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"
cfg.env = "Pendulum-v1"
cfg.total_steps = 1_000_000

agent = DreamerAgent(cfg)
agent.train()
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stoch_size` | 30 | Stochastic latent dimensions |
| `deter_size` | 200 | Deterministic hidden size |
| `embed_size` | 1024 | Encoder embedding size |
| `imagine_horizon` | 15 | Imagination rollout length |
| `discount` | 0.99 | Discount factor γ |
| `td_lambda` | 0.95 | λ-return parameter |
| `kl_loss_coeff` | 1.0 | KL divergence weight |

### Learning Objectives

**World Model Loss**:
```
L_world = L_reconstruction + L_reward + β * L_KL
```

**Actor Loss** (REINFORCE):
```
L_actor = -E[log π(a|s) * (G - V(s))]
```

**Critic Loss** (MSE):
```
L_critic = E[(G - V(s))²]
```

## DreamerV2 Enhancements

DreamerV2 introduces several improvements:

1. **Discrete latents**: Categorical latent variables instead of Gaussian
2. **KL balancing**: Separate weighting for prior/posterior KL
3. **Discount model**: Learns to predict episode termination
4. **Layer normalization**: More stable training

## Environment Support

Dreamer supports multiple backends:

```python
cfg = DreamerConfig()
cfg.env_backend = "dmc"      # DeepMind Control Suite
cfg.env = "walker-walk"

cfg.env_backend = "gym"      # Gym/Gymnasium
cfg.env = "Pendulum-v1"

cfg.env_backend = "unity_mlagents"  # Unity ML-Agents
cfg.unity_file_name = "env.exe"
```

## References

- Hafner, D., Lillicrap, T., Fischer, I., Vuong, Q., Held, D., Haarnoja, T., & Abbeel, P. (2019). Dreamer: Learning Latent Dynamics for Planning from Pixels.
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Mastering Atari with Discrete World Models.
