# Dreamer: Model-Based RL with Latent Dynamics

Dreamer is a model-based reinforcement learning algorithm that learns a latent dynamics model
from images and trains a behavior policy entirely in the latent space.

Based on papers:
- [Dreamer: Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1912.01603) (Hafner et al., 2019)
- [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193) (DreamerV2, Hafner et al., 2020)

```{contents} Contents
:depth: 3
```

## Overview

Dreamer learns a world model from image observations, then trains an actor-critic
policy entirely in the imagination of that world model. No gradients flow from the
environment to the policy — the world model is the only bridge between real
experience and learned behavior.

The family has two major versions:

| Aspect | DreamerV1 | DreamerV2 |
|--------|-----------|-----------|
| Latent type | Gaussian (continuous) | One-hot categorical (discrete) |
| KL balancing | Single coefficient | Separate α for prior/posterior |
| Discount head | Fixed γ | Learned termination predictor |
| Architecture | ELU activations | LayerNorm + SiLU |
| Atari performance | ~40% HNS | ~100% HNS |

## Architecture

### High-level diagram

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

### Detailed architecture (RSSM)

```{mermaid}
graph TD
    A["Image x_t"] --> B["ConvEncoder"]
    B --> C["Obs embed e_t"]
    D["Prev state h_{t-1}, s_{t-1}"] --> E["GRU (deter)"]
    F["Prev action a_{t-1}"] --> E
    E --> G["h_t (deterministic)"]
    G --> H["Prior model p(s_t | h_t)"]
    H --> I["s_t (stochastic prior)"]
    C --> J["Posterior model q(s_t | h_t, e_t)"]
    G --> J
    J --> K["s_t (stochastic posterior)"]
    K --> L["ConvDecoder → x̂_t"]
    K --> M["Reward head → r̂_t"]
    K --> N["Discount head → γ̂_t (V2 only)"]
```

### Recurrent State-Space Model (RSSM)

The core of Dreamer is the RSSM, defined in `world_models.models.dreamer_rssm.RSSM`.
It maintains a hybrid state with two components:

**1. Deterministic state** `h_t` — a GRU hidden state that captures temporal
dependencies and deterministic transitions:

```{math}
h_t = \text{GRU}(h_{t-1}, [s_{t-1}, a_{t-1}])
```

**2. Stochastic state** `s_t` — a latent variable representing uncertainty.
- **V1:** Diagonal Gaussian with `stoch_size` means and variances.
- **V2:** Concatenation of `stoch_size / classes` one-hot categoricals, each
  with `classes` categories. Default: 32 classes × 32 categories = 1024 total.

The model operates in two modes:

**Observe mode** (training — uses real observations):

```{math}
\text{Posterior: } s_t \sim q(s_t | h_t, \text{enc}(x_t))
```

**Imagine mode** (policy training — no observations):

```{math}
\text{Prior: } s_t \sim p(s_t | h_t)
```

Key insight: the prior learns to predict the posterior without seeing the
observation. During imagination, the prior serves as the dynamics model.

### CNN Encoder (`world_models.vision.dreamer_encoder.ConvEncoder`)

Four-layer CNN with increasing channels (32 → 64 → 128 → 256) and ReLU
activations. Strided convolutions (stride 2) halve spatial resolution at
each layer. Output is flattened to `obs_embed_size` (default 1024).

```
Input:  (3, 64, 64)
  └─ Conv2D(3, 32, 4×4, stride 2) → (32, 31, 31)
  └─ Conv2D(32, 64, 4×4, stride 2) → (64, 14, 14)
  └─ Conv2D(64, 128, 4×4, stride 2) → (128, 6, 6)
  └─ Conv2D(128, 256, 4×4, stride 2) → (256, 2, 2)
  └─ Flatten → Linear(1024, embed_size)
Output: embed_size-d vector
```

### CNN Decoder (`world_models.vision.dreamer_decoder.ConvDecoder`)

Mirrored transposed-CNN structure:

```
Input:  stoch + deter state (e.g. 1030-d)
  └─ Linear(1030, 1024) → reshape to (256, 2, 2)
  └─ ConvT2D(256, 128, 5×5, stride 2) → (128, 6, 6)
  └─ ConvT2D(128, 64, 5×5, stride 2) → (64, 14, 14)
  └─ ConvT2D(64, 32, 6×6, stride 2) → (32, 31, 31)
  └─ ConvT2D(32, 3, 6×6, stride 2) → (3, 64, 64)
Output: reconstructed image
```

### Reward and Discount Heads (`DenseDecoder`)

Two-layer MLPs with ELU activations predicting scalar reward, and in V2,
episode discount (termination probability). The discount head is trained with
binary cross-entropy:

```{math}
\mathcal{L}_{\text{disc}} = \text{BCE}(\gamma_t, \hat{\gamma}_t)
```

### Action Decoder (`ActionDecoder`)

Outputs the policy distribution over actions. For continuous actions, predicts
a tanh-squashed Gaussian. For discrete actions, predicts a categorical
distribution. Uses REINFORCE gradient through the world model.

## Training

### World Model Loss

The complete world model objective (V1 notation):

```{math}
\mathcal{L}_{\text{WM}} = \mathcal{L}_{\text{pred}} + \beta \cdot \mathcal{L}_{\text{KL}}
```

Where:

```{math}
\begin{aligned}
\mathcal{L}_{\text{pred}} &=
\underbrace{\|x_t - \hat{x}_t\|^2}_{\text{image reconstruction}}
+ \underbrace{\|r_t - \hat{r}_t\|^2}_{\text{reward prediction}}
+ \underbrace{\text{BCE}(\gamma_t, \hat{\gamma}_t)}_{\text{discount (V2)}} \\
\mathcal{L}_{\text{KL}} &=
D_{\text{KL}}\big(q(s_t | h_t, e_t) \;\|\; p(s_t | h_t)\big)
\end{aligned}
```

In DreamerV2, the KL uses **KL balancing**:

```{math}
\mathcal{L}_{\text{KL}} = \alpha \cdot D_{\text{KL}}[q \| \text{sg}(p)]
+ (1 - \alpha) \cdot D_{\text{KL}}[\text{sg}(q) \| p]
```

where `sg` is stop-gradient and `α` (default 0.8) weights the prior-following
term higher. This prevents the posterior from collapsing.

**Free nats**: A threshold (default 3 nats) below which KL is not penalized.

### Actor-Critic in Imagination

After the world model is trained, Dreamer rolls out imagined trajectories
using the prior dynamics and trains an actor-critic purely in latent space:

```{mermaid}
graph LR
    A["Posterior state s_0"] --> B["Imagine prior p(s_t | h_t)"]
    B --> C[s_1, s_2, ..., s_T]
    C --> D["Reward head → r_t"]
    C --> E["Discount head → γ_t"]
    C --> F["Value head → v_t"]
    D --> G["λ-return G_t"]
    E --> G
    F --> G
    G --> H["Actor: REINFORCE ∇J"]
    G --> I["Critic: MSE(V - G)"]
```

**Actor loss** (REINFORCE with baseline):

```{math}
\mathcal{L}_{\text{actor}} = -\sum_{t=1}^{T}
\log \pi(a_t | s_t) \cdot \text{sg}(G_t^\lambda - V(s_t))
+ \eta \cdot H[\pi(\cdot | s_t)]
```

**Critic loss**:

```{math}
\mathcal{L}_{\text{critic}} = \sum_{t=1}^{T} \|V(s_t) - G_t^\lambda\|^2
```

**Lambda return**:

```{math}
G_t^\lambda = r_t + \gamma_t \cdot
\begin{cases}
(1 - \lambda) V(s_{t+1}) + \lambda G_{t+1}^\lambda & \text{if } t < T \\
V(s_T) & \text{if } t = T
\end{cases}
```

### Training Loop

DreamerAgent follows a cyclic training loop:

1. **Collect**: Interact with environment using current policy (+ exploration noise). Store experience in `ReplayBuffer`.
2. **Train world model** (every step): Sample batch of `batch_size` sequences of length `train_seq_len` from buffer. Update encoder, RSSM, decoder, reward head, and discount head.
3. **Train actor-critic** (every step after `seed_steps`): Imagine `imagine_horizon` steps using prior dynamics. Compute λ-returns. Update actor and critic.
4. **Log**: Metrics, video reconstructions, and checkpointing.

```{math}
\begin{aligned}
&\text{for each environment step:} \\
&\quad \text{collect } (x_t, a_t, r_t, \gamma_t) \\
&\quad \text{if } step > seed\_steps: \\
&\qquad \text{sample batch from buffer} \\
&\qquad \text{update world model (encoder, RSSM, decoder, reward)} \\
&\qquad \text{imagine } H \text{ steps} \\
&\qquad \text{update actor, critic} \\
&\quad \text{log every } log\_every \text{ steps}
\end{aligned}
```

## Usage in TorchWM

### Quick start

```python
import torchwm

agent = torchwm.create_model(
    "dreamer",
    env_backend="dmc",
    env="walker-walk",
    total_steps=5_000_000,
)
agent.train()
```

### Using config directly

```python
from torchwm import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "dmc"
cfg.env = "walker-walk"
cfg.total_steps = 5_000_000

agent = DreamerAgent(cfg)
agent.train()
```

### Environment backends

```python
cfg = DreamerConfig()

cfg.env_backend = "dmc"         # DeepMind Control Suite
cfg.env = "walker-walk"

cfg.env_backend = "gym"         # Gym/Gymnasium
cfg.env = "Pendulum-v1"

cfg.env_backend = "dmlab"       # DeepMind Lab
cfg.env = "rooms_collect_good_objects_train"
cfg.dmlab_action_repeat = 4

cfg.env_backend = "mujoco"      # MuJoCo
cfg.env = "Humanoid-v4"

cfg.env_backend = "brax"        # JAX/Brax
cfg.env = "ant"

cfg.env_backend = "procgen"     # Procgen
cfg.env = "coinrun"

cfg.env_backend = "unity_mlagents"  # Unity ML-Agents
cfg.unity_file_name = "env.exe"
```

### CLI

```bash
torchwm train dreamer --env dmc/walker-walk --device cuda
```

## Config Reference

All configuration is in `world_models.configs.dreamer_config.DreamerConfig`:

```python
from world_models.configs.dreamer_config import DreamerConfig

config = DreamerConfig()

# Dreamer version
config.algo = "Dreamerv1"  # or "Dreamerv2" (default: "Dreamerv1")

# Environment
config.env_backend = "dmc"
config.env = "walker-walk"
config.image_size = (64, 64)

# Model architecture
config.stoch_size = 30
config.deter_size = 200
config.obs_embed_size = 1024

# Training
config.total_steps = 5_000_000
config.batch_size = 50
config.train_seq_len = 50
config.imagine_horizon = 15
config.model_lr = 6e-4

# Actor-critic
config.actor_lr = 8e-5
config.value_lr = 8e-5
config.discount = 0.99
config.td_lambda = 0.95

# KL (V2)
config.kl_alpha = 0.8
config.free_nats = 3.0

# Exploration
config.expl_amount = 0.3
config.expl_decay = 0.0

# Logging
config.log_every = 10_000
config.save_every = 100_000
config.enable_wandb = False
```

### Key Hyperparameters

#### World Model

| Parameter | V1 Default | V2 Default | Effect |
|-----------|------------|------------|--------|
| `stoch_size` | 30 | 32 × 32 classes | Total stochastic capacity |
| `deter_size` | 200 | 200 | GRU hidden size |
| `model_lr` | 6e-4 | 3e-4 | World model learning rate |
| `train_seq_len` | 50 | 50 | Sequence length per batch |
| `batch_size` | 50 | 16 | Sequences per batch |
| `free_nats` | 3.0 | 3.0 | KL free bits threshold |

#### Actor-Critic

| Parameter | V1 Default | V2 Default | Effect |
|-----------|------------|------------|--------|
| `actor_lr` | 8e-5 | 8e-5 | Policy learning rate |
| `value_lr` | 8e-5 | 8e-5 | Critic learning rate |
| `imagine_horizon` | 15 | 15 | Imagination rollout length |
| `discount` | 0.99 | 0.997 | Discount factor |
| `td_lambda` | 0.95 | 0.95 | λ-return parameter |
| `kl_scale` | 1.0 | 1.0 | KL loss coefficient |
| `kl_alpha` | — | 0.8 | KL balancing weight (V2 only) |

#### Environment Interaction

| Parameter | Default | Effect |
|-----------|---------|--------|
| `action_repeat` | 2 | Repeat each action N times |
| `action_noise` | 0.3 | Exploration noise std |
| `seed_steps` | 5000 | Random steps before training |
| `total_steps` | 5e6 | Total environment steps |
| `collect_steps` | 1000 | Steps between model updates |

## DreamerV1 vs DreamerV2

### Categorical vs Gaussian Latents

V1 uses a diagonal Gaussian for the stochastic state. V2 uses a concatenation
of one-hot categorical distributions, each independently sampled:

```python
# V1: single Gaussian
self.stoch = torch.distributions.Normal(mean, std)

# V2: stack of categoricals (e.g., 32 classes × 32 categories)
self.stoch = torch.cat([one_hot(logits[i]) for i in range(num_categories)], dim=-1)
```

### KL Balancing

V1 applies a single KL coefficient `β`:

```{math}
\mathcal{L}_{\text{KL}} = β \cdot D_{\text{KL}}[q \| p]
```

V2 splits the KL with stop-gradient:

```{math}
\mathcal{L}_{\text{KL}} = α \cdot D_{\text{KL}}[q \| \text{sg}(p)]
+ (1 - α) \cdot D_{\text{KL}}[\text{sg}(q) \| p]
```

### Discount Head

V2 learns to predict episode termination (`γ̂_t`) via a binary classifier,
critical for Atari where episodes can end due to life loss.

### Architecture Improvements

- **Layer normalization** in GRU and MLP layers (V2)
- **SiLU activations** replace ELU (V2)
- **Two-hot reward encoding** replaces MSE (V2): discretizes reward into 255 bins

## Common Pitfalls

### Posterior collapse

If the stochastic state is ignored by the dynamics, the model reduces to a
deterministic RNN. Symptoms: good reconstruction but imagination diverges.

**Fixes:**
- Increase `kl_scale` (V1) or adjust `kl_alpha` (V2)
- Decrease `free_nats`
- Reduce `stoch_size`

### Imagination divergence

The prior predicts states that drift from realistic latents over long horizons.

**Fixes:**
- Keep `imagine_horizon` short (10–15)
- Verify multi-step prediction, not just one-step

### NaN loss during training

**Fixes:**
- Reduce `model_lr` to 1e-4
- Tighten gradient clipping (default 100 → 10)
- Enable layer norm

### Actor never improves

**Fixes:**
- Increase `imagine_horizon` for delayed rewards
- Increase entropy bonus via `expl_amount`
- Verify critic loss is decreasing

## References

- Hafner, D., Lillicrap, T., Fischer, I., Vuong, Q., Held, D., Haarnoja, T., & Abbeel, P. (2019). Dreamer: Learning Latent Dynamics for Planning from Pixels.
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Mastering Atari with Discrete World Models.
- Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). Mastering Atari with Discrete World Models (DreamerV2). *ICLR 2021.*
