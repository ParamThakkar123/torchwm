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

```{mermaid}
graph TD
    subgraph WM["World Model RSSM"]
        A["Encoder CNN 64x64"] --> B["Latent model GRU and stochastic state"]
        B --> C["Decoder transposed CNN"]
        B --> D["Prior stochastic state"]
        E["Recurrent state update"] --> B
    end

    subgraph IR["Imagination Rollout"]
        F["Latent state 0"] --> G["Action 0"]
        G --> H["Latent state 1"]
        H --> I["Action 1"]
        I --> J["Latent state 2"]
        J --> K["Future steps"]
        K --> L["Latent horizon state"]
        L --> M["Lambda return target"]
    end

    subgraph AC["Actor Critic Learning"]
        N["Actor policy with baseline"]
        O["Critic value model"]
    end

    C --> F
    M --> N
    M --> O

    style A fill:#e1f5fe
    style N fill:#e8f5e8
    style O fill:#e8f5e8
```
## Components

### 1. Recurrent State Space Model (RSSM)

The core world model combining:
- **Deterministic hidden state** (h_t): Recurrent state (GRU)
- **Stochastic latent state** (s_t): Discrete or continuous latent variables

**Dynamics**:

```{math}
\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{s}_{t-1}, \mathbf{a}_{t-1})
```

**Posterior**:

```{math}
\mathbf{s}_t \sim q(\mathbf{s}_t | \mathbf{h}_t, \mathbf{x}_t)
```

**Prior**:

```{math}
\mathbf{s}_t \sim p(\mathbf{s}_t | \mathbf{h}_t)
```

### 2. Encoder/Decoder

- **Encoder**: CNN that maps images to latent embeddings
- **Decoder**: Transposed CNN that reconstructs images from latents
- Both use ReLU activations and residual connections

### 3. Reward/Discount Heads

- **Reward model**: Predicts reward from latent state
- **Discount model**: Predicts episode termination (DreamerV2)

## Training

```python :class: thebe
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

```{math}
\mathcal{L}_\mathrm{world} = \mathcal{L}_\mathrm{reconstruction} + \mathcal{L}_\mathrm{reward} + \beta \cdot \mathcal{L}_\mathrm{KL}
```

**Actor Loss** (REINFORCE):

```{math}
\mathcal{L}_\mathrm{actor} = -\mathbb{E}[\log \pi(\mathbf{a} | \mathbf{s}) \cdot (G - V(\mathbf{s}))]
```

**Critic Loss** (MSE):

```{math}
\mathcal{L}_\mathrm{critic} = \mathbb{E}[(G - V(\mathbf{s}))^2]
```

## DreamerV2 Enhancements

DreamerV2 introduces several improvements:

1. **Discrete latents**: Categorical latent variables instead of Gaussian
2. **KL balancing**: Separate weighting for prior/posterior KL
3. **Discount model**: Learns to predict episode termination
4. **Layer normalization**: More stable training

## Environment Support

Dreamer supports multiple backends:

```python :class: thebe
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
