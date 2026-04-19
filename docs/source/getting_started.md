# Getting Started

## Installation

Install from PyPI:

```bash
pip install torchwm
```

Install from source:

```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Logging with Weights & Biases and TensorBoard

TorchWM supports logging experiment results to Weights & Biases (WandB) and TensorBoard.

### Weights & Biases

To use WandB logging, you must provide an API key as anonymous logins are no longer supported.

1. Get your WandB API key from [wandb.ai](https://wandb.ai/settings).
2. Set the key in your config:

```python
cfg.enable_wandb = True
cfg.wandb_api_key = "your-api-key-here"
cfg.wandb_project = "torchwm"
cfg.wandb_entity = "your-entity"
```

### TensorBoard

Enable TensorBoard logging:

```python
cfg.enable_tensorboard = True
cfg.log_dir = "runs"
```

Logs will be saved to the specified directory and can be viewed with `tensorboard --logdir runs`.

## Quick Start: Dreamer

TorchWM implements multiple world model algorithms. Click on each to see detailed documentation:

| Algorithm | Description | Quick Start |
|-----------|-------------|--------------|
| **Dreamer** | Model-based RL with latent dynamics | {doc}`dreamer` |
| **JEPA** | Self-supervised visual representations | {doc}`jepa` |
| **IRIS** | Sample-efficient RL with Transformers | {doc}`iris` |
| **DiT** | Diffusion models with Transformers | {doc}`dit` |

## Quick Start: Inference with Operators

TorchWM now includes standardized operators for preprocessing inputs during inference, making it easy to deploy models consistently.

### What are Operators?

Operators handle input preprocessing: normalizing images, encoding actions, tokenizing text, and generating masks. Each model has a dedicated operator that ensures inputs are in the correct format.

### Basic Usage

```python
from world_models.inference.operators import DreamerOperator

# Create operator with config parameters
op = DreamerOperator(
    image_size=64,  # Image size for Dreamer
    action_dim=6    # Action dimension
)

# Process raw inputs
raw_inputs = {
    'image': your_pil_image_or_tensor,  # PIL Image or torch.Tensor
    'action': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # Action as list
}

# Get processed tensors
processed = op.process(raw_inputs)
# Returns: {'obs': tensor(B, 3, 64, 64), 'action': tensor(B, 6)}
```

### Available Operators

| Operator | Model | Purpose | Key Parameters |
|----------|-------|---------|----------------|
| `DreamerOperator` | Dreamer | Image/action preprocessing | `image_size`, `action_dim` |
| `JEPAOperator` | JEPA | Image masking and patching | `image_size`, `patch_size`, `mask_ratio` |
| `IrisOperator` | IRIS | Sequence tokenization | `seq_length`, `vocab_size` |
| `PlaNetOperator` | PlaNet | State/action transitions | `state_dim`, `action_dim` |

### JEPA Example (Self-Supervised)

```python
from world_models.inference.operators import JEPAOperator

op = JEPAOperator(image_size=224, patch_size=16, mask_ratio=0.75)
inputs = {'images': [image1, image2]}
result = op(inputs)
# result['images']: stacked normalized images
# result['mask']: random mask for self-supervised learning
```

### IRIS Example (Sequence Processing)

```python
from world_models.inference.operators import IrisOperator

op = IrisOperator(seq_length=512, vocab_size=32000)
inputs = {'tokens': [101, 2054, 2003, 102]}  # Token sequence
result = op(inputs)
# result['input_ids']: padded token tensor
# result['attention_mask']: attention mask
```

### Integration with Configs

Operators use parameters from config classes:

```python
from world_models.configs import DreamerConfig

cfg = DreamerConfig()
op = DreamerOperator(
    image_size=cfg.operator_image_size,
    action_dim=cfg.operator_action_dim
)
```

### Utilities

Common preprocessing functions are available in `world_models.inference.operators.utils`:

```python
from world_models.inference.operators.utils import normalize_image, tokenize_text

normalized_img = normalize_image(pil_image, size=(224, 224))
tokens = tokenize_text("Hello world", max_length=512)
```

## Environment Backends

Dreamer supports multiple backends through `DreamerConfig.env_backend`:

- `dmc`: DeepMind Control Suite tasks (for example `walker-walk`)
- `gym`: Gym/Gymnasium environment IDs or an existing environment instance
- `unity_mlagents`: Unity ML-Agents executable environments

## Typical Training Flow

1. Choose an algorithm (Dreamer, JEPA, IRIS, or DiT)
2. Create a config object for that algorithm
3. Override dataset/environment and optimization fields
4. Instantiate the corresponding agent
5. Call `train()` and monitor logs/checkpoints

For complete API details, see {doc}`api_reference`.
