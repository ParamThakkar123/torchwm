# IRIS: Transformers for Sample-Efficient World Models

IRIS (Imagination with auto-Regression over an Inner Speech) is an implementation of the paper
"Transformers are Sample-Efficient World Models" (Micheli et al., 2023).

## Key Idea

IRIS achieves **human-level performance on Atari with only ~2 hours of gameplay** (100k environment steps)
by learning entirely in the imagination of a world model:

1. **Train world model** from real interactions
2. **Generate imagined trajectories** in the latent space
3. **Train policy** purely on imagined data

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Discrete Autoencoder                      │
│  ┌─────────┐    ┌────────────┐    ┌─────────────┐               │
│  │ Encoder │ -> │   VQVAE    │ -> │   Decoder   │               │
│  │ (CNN)   │    │ (512 vocab)│    │ (Transposed │               │
│  │ 64x64   │    │  16 tokens │    │   CNN)      │               │
│  └─────────┘    └────────────┘    └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Autoregressive Transformer                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  z_t (16 tokens) ──► a_t ──► z_{t+1} (16 tokens)         │   │
│  │         ↓              ↓              ↓                  │   │
│  │      Reward        Termination     Reward...             │   │
│  └──────────────────────────────────────────────────────────┘   │
│  GPT-style: 10 layers, 4 heads, 256 embedding dim              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Actor-Critic in Imagination                  │
│  ┌─────────────┐    ┌──────────────────┐                       │
│  │   Actor     │    │     Critic       │                       │
│  │ (CNN+LSTM)  │    │  (CNN+LSTM)      │                       │
│  │             │    │                  │                       │
│  │ λ-return    │    │   MSE loss       │                       │
│  │ REINFORCE   │    │                  │                       │
│  └─────────────┘    └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Discrete Autoencoder (VQVAE)

- **Encoder**: 4-layer CNN with residual blocks and self-attention
- **Quantization**: Vector Quantizer with EMA updates (512 codebook size)
- **16 tokens per frame**: Reduces sequence length for Transformer efficiency
- **Loss**: L1 reconstruction + commitment loss

### 2. Transformer World Model

- **GPT-style architecture**: 10 layers, 4 attention heads, 256 embedding dim
- **Autoregressive**: Predicts next tokens one-by-one
- **Heads**: Token prediction, reward prediction, termination prediction
- **Self-supervised training**: Cross-entropy on token sequences

### 3. Actor-Critic

- **CNN + LSTM**: Processes reconstructed frames
- **λ-returns**: Balances bias and variance in value estimation
- **REINFORCE**: Policy gradient with baseline
- **Entropy bonus**: Maintains exploration

## Training

```python
from world_models.training.train_iris import IRISTrainer
from world_models.configs.iris_config import IRISConfig

trainer = IRISTrainer(
    game="ALE/Pong-v5",
    device="cuda",
    seed=42,
)

trainer.train(total_epochs=600)
```

### Warm-start Delays

| Component | Start Epoch | Description |
|-----------|-------------|-------------|
| Autoencoder | 5 | Learn frame compression first |
| Transformer | 25 | Learn dynamics once tokens are good |
| Actor-Critic | 50 | Learn policy in imagination |

### Key Hyperparameters

- **Frame size**: 64x64
- **Tokens per frame**: 16 (from 512 vocabulary)
- **Transformer sequence length**: 20 timesteps
- **Imagination horizon**: 20 steps
- **Discount (γ)**: 0.995
- **λ for λ-return**: 0.95

## Benchmark Results

| Metric | IRIS (ours) | SPR | DrQ | CURL | SimPLe |
|--------|-------------|-----|-----|------|--------|
| Mean HNS | **1.046** | 0.616 | 0.465 | 0.261 | 0.332 |
| Superhuman games | **10/26** | 6/26 | 3/26 | 2/26 | 1/26 |

## Usage

### Training

```bash
# Single game
python -m world_models.training.train_iris --game "ALE/Pong-v5"

# Benchmark
python -m benchmarks.atari_100k --device cuda --num_seeds 5
```

### Configuration

```python
from world_models.configs.iris_config import IRISConfig

config = IRISConfig()

# Autoencoder
config.vocab_size = 512
config.tokens_per_frame = 16

# Transformer
config.transformer_layers = 10
config.transformer_embed_dim = 256

# Training
config.total_epochs = 600
config.env_steps_per_epoch = 200
```

## References

- Micheli, Vincent, Eloi Alonso, and François Fleuret. "Transformers are Sample-Efficient World Models." ICLR 2023.
