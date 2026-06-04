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

<div class="architecture-diagram" aria-label="Dreamer architecture diagram">
  <section class="diagram-section">
    <h3>World Model RSSM</h3>
    <div class="diagram-row">
      <span class="diagram-node info">Encoder CNN 64x64</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">GRU plus stochastic latent model</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Decoder transposed CNN</span>
    </div>
  </section>
  <section class="diagram-section">
    <h3>Imagination Rollout</h3>
    <div class="diagram-row">
      <span class="diagram-node">State s0</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Action a0</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Imagined future states</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Lambda-return target</span>
    </div>
  </section>
  <section class="diagram-section">
    <h3>Actor-Critic Learning</h3>
    <div class="diagram-row">
      <span class="diagram-node success">Actor policy</span>
      <span class="diagram-node success">Critic value model</span>
    </div>
  </section>
</div>

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
\mathbf{s}_t \sim q(\mathbf{s}_t \mid \mathbf{h}_t, \mathbf{x}_t)
```

**Prior**:

```{math}
\mathbf{s}_t \sim p(\mathbf{s}_t \mid \mathbf{h}_t)
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
\begin{aligned}
\mathcal{L}_\mathrm{world}
&= \mathcal{L}_\mathrm{reconstruction}
 + \mathcal{L}_\mathrm{reward}
 + \beta \cdot \mathcal{L}_\mathrm{KL}
\end{aligned}
```

**Actor Loss** (REINFORCE):

```{math}
\mathcal{L}_\mathrm{actor}
= -\mathbb{E}\left[\log \pi(\mathbf{a} \mid \mathbf{s}) \cdot (G - V(\mathbf{s}))\right]
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

cfg.env_backend = "brax"     # JAX/Brax
cfg.env = "ant"
cfg.brax_backend = "generalized"

cfg.env_backend = "unity_mlagents"  # Unity ML-Agents
cfg.unity_file_name = "env.exe"
```

## References

- Hafner, D., Lillicrap, T., Fischer, I., Vuong, Q., Held, D., Haarnoja, T., & Abbeel, P. (2019). Dreamer: Learning Latent Dynamics for Planning from Pixels.
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Mastering Atari with Discrete World Models.
