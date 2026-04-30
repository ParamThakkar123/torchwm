# Using Operators for Inference

This guide explains how to use TorchWM's inference operators for standardized input preprocessing.

## Overview

Operators provide a consistent interface for preprocessing inputs before feeding them to world models. Each model has a dedicated operator that handles:

- Image normalization and resizing
- Action encoding and formatting
- Sequence tokenization and padding
- Mask generation for self-supervised tasks

## Base Operator Class

All operators inherit from `OperatorABC`:

```python :class: thebe
from world_models.inference.operators.base import OperatorABC

class MyOperator(OperatorABC):
    def process(self, inputs):
        # Your preprocessing logic
        return processed_tensors
```

## Dreamer Operator

For Dreamer model's image and action processing:

```python :class: thebe
from world_models.inference.operators import DreamerOperator
from PIL import Image
import torch

op = DreamerOperator(image_size=64, action_dim=6)

# Process single image and action
image = Image.open('obs.png')
action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

result = op.process({'image': image, 'action': action})
print(result['obs'].shape)    # torch.Size([1, 3, 64, 64])
print(result['action'].shape) # torch.Size([1, 6])

# Process tensor inputs
obs_tensor = torch.randn(3, 64, 64)
result = op.process({'image': obs_tensor, 'action': torch.tensor(action)})
```

## JEPA Operator

For JEPA's self-supervised image processing with masking:

```python :class: thebe
from world_models.inference.operators import JEPAOperator

op = JEPAOperator(image_size=224, patch_size=16, mask_ratio=0.75)

# Process batch of images
images = [Image.open(f'img_{i}.jpg') for i in range(4)]
result = op.process({'images': images})

print(result['images'].shape)  # torch.Size([4, 3, 224, 224])
print(result['mask'].shape)    # torch.Size([196]) - flattened patch mask

# Custom mask
custom_mask = torch.ones(196)  # 14x14 patches = 196
result = op.process({'images': images, 'mask': custom_mask})
```

## IRIS Operator

For IRIS's sequence processing:

```python :class: thebe
from world_models.inference.operators import IrisOperator

op = IrisOperator(seq_length=512, vocab_size=32000)

# Process token sequence
tokens = [101, 7592, 1010, 2088, 102]  # Example tokens
result = op.process({'tokens': tokens})

print(result['input_ids'].shape)      # torch.Size([1, 512])
print(result['attention_mask'].shape) # torch.Size([1, 512])

# Process with embeddings
embeddings = torch.randn(1, 768)
result = op.process({'tokens': tokens, 'embeddings': embeddings})
```

## PlaNet Operator

For PlaNet's environment state processing:

```python :class: thebe
from world_models.inference.operators import PlaNetOperator

op = PlaNetOperator(state_dim=32, action_dim=4)

# Process transition data
inputs = {
    'obs': torch.randn(32),  # State vector
    'action': [0.1, 0.2, 0.3, 0.4],
    'reward': 1.0,
    'done': False
}

result = op.process(inputs)
print(result['obs'].shape)    # torch.Size([1, 32])
print(result['action'].shape) # torch.Size([1, 4])
print(result['reward'].shape) # torch.Size([1])
print(result['done'].shape)   # torch.Size([1])
```

## Configuration Integration

Operators work seamlessly with config classes:

```python :class: thebe
from world_models.configs import DreamerConfig, JEPAConfig, IRISConfig

# Dreamer
dreamer_cfg = DreamerConfig()
dreamer_op = DreamerOperator(
    image_size=dreamer_cfg.operator_image_size,
    action_dim=dreamer_cfg.operator_action_dim
)

# JEPA
jepa_cfg = JEPAConfig()
jepa_op = JEPAOperator(
    image_size=jepa_cfg.operator_image_size,
    patch_size=jepa_cfg.operator_patch_size,
    mask_ratio=jepa_cfg.operator_mask_ratio
)

# IRIS
iris_cfg = IRISConfig()
iris_op = IrisOperator(
    seq_length=iris_cfg.operator_seq_length,
    vocab_size=iris_cfg.operator_vocab_size
)
```

## Utilities

Common preprocessing functions in `world_models.inference.operators.utils`:

```python :class: thebe
from world_models.inference.operators.utils import (
    normalize_image,
    tokenize_text,
    resize_image
)

# Image processing
normalized = normalize_image(pil_image, size=(224, 224))
resized = resize_image(tensor_image, size=(64, 64))

# Text processing
tokens = tokenize_text("Hello world", max_length=512)
```

## Error Handling

Operators validate inputs and provide helpful error messages:

```python :class: thebe
try:
    result = op.process(invalid_inputs)
except ValueError as e:
    print(f"Preprocessing error: {e}")
```

## Best Practices

1. **Use with configs**: Always instantiate operators using config parameters for consistency.
2. **Batch processing**: Operators handle both single inputs and batches.
3. **Device placement**: Processed tensors are on CPU; move to GPU as needed.
4. **Type checking**: Operators accept various input types (PIL, tensors, lists) and standardize them.
5. **Performance**: Operators use optimized transforms for fast preprocessing.

## Future: Pipelines

Operators are designed to integrate with future inference pipelines, providing the preprocessing layer in a standardized Pipeline + Operator + Task architecture.