# Genie: Generative Interactive Environment

Genie is a generative model trained from video-only data that can be used as an
interactive environment for reinforcement learning and decision-making tasks.

Based on paper: [Genie: Generative Interactive Environments](https://arxiv.org/abs/2406.15114) (Bruce et al., 2024)

## Key Idea

Genie learns to understand world dynamics from unlabeled videos by learning:

1. **Video Tokenization**: Converts raw video frames into discrete tokens
2. **Latent Actions**: Infers the underlying actions that caused transitions between frames
3. **Dynamics Prediction**: Predicts future frames given past frames and latent actions

This enables agents to imagine and plan in a learned latent action space without
needing explicit action labels.

## Architecture

<div class="architecture-diagram" aria-label="Genie architecture diagram">
  <section class="diagram-section">
    <h3>Genie Architecture</h3>
    <div class="diagram-row">
      <span class="diagram-node">Video frames</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node info">Video tokenizer</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Video tokens</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node success">Dynamics model</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node warning">Decoded frames</span>
    </div>
    <div class="diagram-row">
      <span class="diagram-node">Consecutive frames</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node info">Latent action model</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Latent actions</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node success">Dynamics model</span>
    </div>
  </section>
</div>

## Components

### 1. Video Tokenizer

Converts raw video frames into discrete tokens using a VQ-VAE approach:

- Encoder processes frames into latent representations
- Quantization layer maps latents to discrete codebook indices
- Decoder reconstructs video from discrete tokens

```
Input: (B, C, T, H, W) → Tokens: (B, T, H/patch, W/patch)
```

### 2. Latent Action Model (LAM)

Learns to infer latent actions from video帧 transitions:

- Encodes pairs of consecutive frames
- Predicts discrete latent action tokens
- Uses VQ commitment loss for stable training

```
Input: (Frame_t, Frame_t+1) → Latent Action Index ∈ {0, ..., V-1}
```

### 3. Dynamics Model

Transformer-based model that predicts future tokens:

- Autoregressive generation of video tokens
- Conditioned on latent actions
- Uses MaskGIT for efficient sampling

## Training

```python :class: thebe
from world_models.models import create_genie_small
from world_models.configs import GenieConfig
from world_models.training import GenieTrainer

cfg = GenieConfig()
cfg.num_frames = 16
cfg.image_size = 64
cfg.epochs = 100

model = create_genie_small(num_frames=16, image_size=64)
trainer = GenieTrainer(model, cfg)
trainer.train()
```

### Training Losses

Genie is trained with multiple loss components:

1. **Tokenizer Loss**: VQ-VAE reconstruction loss for video tokenization
2. **Latent Action Loss**: VQ commitment loss + prediction loss for action learning
3. **Dynamics Loss**: Cross-entropy for token prediction with masking

```
Total Loss = L_tokenizer + λ₁·L_action + λ₂·L_dynamics
```

### Data Format

Prepare videos as a dataset with the following structure:

```
Dataset/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
```

Each video should contain at least `num_frames` frames. The tokenizer will sample
frames uniformly from each video during training.

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_frames` | 8 | Number of frames per video |
| `image_size` | 32 | Input image size (height/width) |
| `tokenizer_vocab_size` | 1024 | Video token vocabulary size |
| `action_vocab_size` | 8 | Latent action vocabulary size |
| `dynamics_dim` | 512 | Transformer hidden dimension |
| `dynamics_depth` | 8 | Number of transformer layers |
| `dynamics_num_heads` | 8 | Number of attention heads |
| `batch_size` | 4 | Training batch size |
| `learning_rate` | 3e-5 | Learning rate |
| `maskgit_steps` | 25 | Number of MaskGIT sampling steps |
| `warmup_steps` | 5000 | Learning rate warmup steps |
| `max_steps` | 125000 | Total training steps |

## Usage

### Generation

Generate new video frames from a prompt frame:

```python
prompt_frame = torch.randn(1, 3, 64, 64)
generated = model.generate(prompt_frame, num_frames=16)
```

### Interactive Play

Step through the environment using inferred or specified actions:

```python
current_frame = torch.randn(1, 3, 64, 64)
action = torch.tensor([3])  # Latent action index (tensor)
next_frame = model.play(current_frame, action)
```

### Action Inference

Infer latent actions from real video frames:

```python
frames = torch.randn(1, 3, 16, 64, 64)
actions = model.infer_actions(frames)
```

## Model Variants

| Model | Parameters | Use Case |
|-------|------------|----------|
| `create_genie_small` | ~50M | Development/testing |
| `create_genie_large` | ~11B | Production/research |

## Comparison to Other Methods

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| JEPA | Images | Latent predictions | Representation learning |
| IRIS | Images | Token sequences | World modeling |
| Genie | Videos | Interactive env | RL agent training |

## Related Components

Genie is built from several core components in this library:

- [`VideoTokenizer`](operators_guide.md#videotokenizer): Encodes video into discrete tokens
- [`LatentActionModel`](operators_guide.md#latent-action-model): Learns latent actions from frame pairs
- [`DynamicsModel`](operators_guide.md#dynamics-model): Transformer for future token prediction

## Advantages

1. **Video-only training**: No action labels required
2. **Interactive**: Can be used as a simulated environment
3. **Generalizable**: Learns from diverse video data
4. **Latent action space**: Enables efficient planning

## References

- Bruce, J., et al. (2024). Genie: Generative Interactive Environments.
