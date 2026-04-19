# JEPA: Joint Embedding Predictive Architecture

JEPA is a self-supervised learning method that learns visual representations by predicting
future representations in latent space, without relying on generative modeling.

Based on paper: [JEPA: Joint Embedding Predictive Architecture](https://arxiv.org/abs/2205.14221) (Bardes et al., 2022)

## Key Idea

Instead of predicting pixels (like autoencoders) or reconstructing images, JEPA predicts
**latent representations** of future frames from past frames:

1. **Encoder**: Encodes current frame into latent space
2. **Predictor**: Predicts future latent representation
3. **Loss**: MSE between predicted and actual future latents

This approach avoids the complexity of pixel-level generation while learning rich representations.

## Architecture

.. mermaid::

    graph TD
        subgraph "JEPA Architecture"
            A[Frame t] --> B[Enc_s<br/>Target Encoder]
            C[Frame t+k] --> D[Enc_s<br/>Target Encoder<br/>Frozen]
            B --> E[Target<br/>z_t]
            D --> E
            F[Predictor<br/>token] --> G[Predict<br/>z_t']
            G --> H[Loss<br/>||z_t' - z_t||²]
            E --> H
        end
        
        style B fill:#fff3cd
        style D fill:#fff3cd
        style H fill:#f8d7da

## Components

### 1. Encoder (Target Encoder)

Encodes input images into latent representations. Two variants available:

- **Spatial Encoder**: Processes full images, preserves spatial structure
- **Temporal Encoder**: Processes video sequences, captures motion

### 2. Predictor

Predicts future latent representations from current encoding:
- Uses masked prediction (similar to masked autoencoders)
- Can use spatial/temporal masking strategies
- Architecture: ViT-style transformer with masked prediction head

### 3. Loss Functions

**Main Loss**: MSE between predicted and target representations
```
L = ||predict(target) - target||²
```

**Additional Losses** (configurable):
- Variance/Invariance regularization
- Cross-view consistency for multi-frame input

## Training

```python :class: thebe
from world_models.models import JEPAAgent
from world_models.configs import JEPAConfig

cfg = JEPAConfig()
cfg.dataset = "imagefolder"
cfg.root_path = "./data"
cfg.image_folder = "train"
cfg.epochs = 100

agent = JEPAAgent(cfg)
agent.train()
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 224 | Input image size |
| `patch_size` | 16 | ViT patch size |
| `embed_dim` | 768 | Embedding dimension |
| `num_layers` | 12 | Transformer layers |
| `num_heads` | 12 | Attention heads |
| `mask_ratio` | 0.75 | Fraction of patches to mask |
| `predict_horizon` | 1 | Frames to predict ahead |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 1e-4 | Learning rate |

## Masking Strategies

JEPA supports multiple masking approaches:

1. **Block masking**: Random rectangular regions
2. **Random masking**: Random individual patches
3. **Temporal masking**: Mask future frames (for video)

```
┌─────────────────────────────────────────────────────┐
│               Masking Strategies                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Original    Block Mask    Random Mask   Temporal  │
│  ┌───┬───┐  ┌───┬───▓──┐  ┌─▓─┬─▓─┼▓─┐  ┌───┬───┐  │
│  │ A │ B │  │ A │  C   │  │ A │ B │ C │  │ A │ B │  │
│  ├───┼───┤  ├───┼───▓──┤  ├─▓─┼─▓─┼▓─┤  ├───┼─▓─┤  │
│  │ C │ D │  │ C │  D   │  │ C │ D │ A │  │ C │ D │  │
│  └───┴───┘  └───┴───▓──┘  └───▓─┴─▓─┴─▓─┘  └───┴─▓─┘  │
│                                                     │
│  Predict C   Predict B     Predict BCD    Predict C │
│  from A,B    from A        from A         from A,B │
└─────────────────────────────────────────────────────┘
```

## Uses for Learned Representations

JEPA representations can be used for:

1. **Downstream tasks**: Fine-tune for classification/detection
2. **World models**: Use as encoder for model-based RL
3. **Planning**: Predict future states for MPC/trajectory optimization
4. **Representation learning**: Pre-train for transfer

## Comparison to Other Methods

| Method | What it predicts | Approach |
|--------|------------------|----------|
| Autoencoder | Pixels | Reconstruction |
| VAE | Pixels | Generative |
| MAE | Pixels | Masked modeling |
| JEPA | Latents | Predictive coding |
| IRIS | Tokens | Transformer dynamics |

## References

- Bardes, A., Ponce, J., & LeCun, Y. (2022). JEPA: Joint Embedding Predictive Architecture.
