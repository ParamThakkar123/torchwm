# Tutorial: JEPA - Joint Embedding Predictive Architecture

JEPA (Joint Embedding Predictive Architecture) is a self-supervised learning
method that learns representations by predicting one view of data from another.
It's particularly effective for visual representations.

## What is JEPA?

JEPA learns by predicting latent representations rather than reconstructing
raw pixels:

```{figure} /_static/jepa_architecture.png
:alt: JEPA Architecture
:width: 600px

JEPA predicts embeddings rather than pixel-space outputs
```

## The JEPA Objective

### Masked Prediction

JEPA uses a masking strategy where some portions of the input are masked
and the model predicts them from context:

```{math}
\mathcal{L}_{\text{JEPA}} = \mathbb{E}_{(x, y) \sim D}[ \| f_\theta(y) - \text{sg}[f_\phi(x)] \|^2 ]
```

Where:
- :math:`x` = context view (partially visible)
- :math:`y` = target view (to be predicted)
- :math:`f_\theta` = predictor network
- :math:`f_\phi` = target encoder (stop-gradient)
- :math:`\text{sg}` = stop-gradient operator

### Why Predict Embeddings?

1. **Avoids trivial solutions**: Pixel reconstruction can collapse to identity
2. **Semantic representations**: Learns meaningful features implicitly
3. **Efficient**: Works with partial observations

## Implementation in TorchWM

### Basic Usage

```python
from world_models.configs import JEPAConfig
from world_models.models import JEPAAgent

cfg = JEPAConfig()
cfg.dataset = "imagefolder"
cfg.root_path = "./data"
cfg.image_folder = "train"
cfg.epochs = 100
cfg.batch_size = 64
cfg.embed_dim = 384
cfg.predictor_depth = 6

agent = JEPAAgent(cfg)
agent.train()
```

### Using Built-in Datasets

```python
# CIFAR-10
cfg.dataset = "cifar10"
cfg.data_dir = "./data"

# ImageNet-1K
cfg.dataset = "imagenet1k"
cfg.data_dir = "./data/imagenet"
```

## Deep Dive: The Mathematics

### Encoder Architecture

```python
class VisionEncoder(nn.Module):
    """ViT-based encoder for JEPA
    
    Converts images to patch embeddings and passes through Transformer
    """
    def __init__(self, image_size=224, patch_size=16, embed_dim=384, depth=12):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position encoding
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=6,
            dim_feedforward=4 * embed_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Encode
        x = self.transformer(x)
        
        return x  # Return all tokens or just cls_token
```

### Predictor Network

```python
class Predictor(nn.Module):
    """Predictor network - predicts target from context
    
    Predictor receives context encoding and predicts target representations
    """
    def __init__(self, embed_dim=384, predictor_depth=6, num_mask_tokens=64):
        super().__init__()
        
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_mask_tokens, embed_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=6,
            dim_feedforward=4 * embed_dim,
            batch_first=True
        )
        self.predictor_transformer = nn.TransformerEncoder(
            encoder_layer, predictor_depth
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, context_enc, mask_positions):
        """
        Args:
            context_enc: (B, context_len, embed_dim) - context encoding
            mask_positions: (B, num_masks) - mask position indices
        """
        # Create queries for mask positions
        B = context_enc.size(0)
        queries = self.predictor_pos_embed[:, :mask_positions.size(1), :]
        
        # Expand context to match batch size
        context_enc = context_enc.expand(mask_positions.size(1), -1, -1)
        
        # Predictor transformer
        predictions = self.predictor_transformer(queries, context_enc)
        
        return self.output_proj(predictions)
```

### Multi-Block Masking

```python
class MaskCollator:
    """Multi-block masking strategy - masks multiple non-overlapping regions"""
    
    def __init__(self, num_mask_blocks=4, min_block_size=16, max_block_size=32):
        self.num_mask_blocks = num_mask_blocks
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
    
    def __call__(self, images):
        """
        Args:
            images: (B, C, H, W)
        Returns:
            context: image with context regions
            target: full original image  
            mask: mask tensor
        """
        B, C, H, W = images.shape
        
        # Create initial full mask
        mask = torch.ones(B, H, W, dtype=torch.bool)
        
        # Generate random blocks for each sample
        for b in range(B):
            for _ in range(self.num_mask_blocks):
                h = torch.randint(self.min_block_size, self.max_block_size, (1,))
                w = torch.randint(self.min_block_size, self.max_block_size, (1,))
                
                # Random position
                y = torch.randint(0, H - h.item() + 1, (1,))
                x = torch.randint(0, W - w.item() + 1, (1,))
                
                # Apply block mask
                mask[b, y:y+h, x:x+w] = False
        
        # Create context and target views
        context = images.clone()
        context[mask] = 0  # Zero out masked regions
        
        return context, images, mask
```

### Complete JEPA Loss

```python
def jepa_loss(predictions, targets, predictor, target_encoder):
    """JEPA loss function
    
    L = || predictor(context) - stopgrad(target_encoder(target)) ||^2
    """
    # Target encoding (stop gradient)
    with torch.no_grad():
        target_features = target_encoder(targets)
    
    # Predict
    predicted_features = predictions
    
    # MSE loss
    loss = F.mse_loss(predicted_features, target_features)
    
    return loss
```

## Training with Custom Data

```python
import torch
from torch.utils.data import Dataset, DataLoader
from world_models.configs import JEPAConfig
from world_models.models import JEPAAgent
from world_models.transforms import make_jepa_transforms

class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Your data loading logic
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image

# Configuration
cfg = JEPAConfig()
cfg.epochs = 100
cfg.lr = 1e-4
cfg.weight_decay = 0.05
cfg.embed_dim = 384
cfg.predictor_depth = 6

# Create dataset and dataloader
transform = make_jepa_transforms(image_size=224)
dataset = CustomImageDataset("./my_data", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Train
agent = JEPAAgent(cfg)
agent.train(dataloader=dataloader)
```

## Using Pretrained JEPA Representations

```python
from world_models.models import JEPAAgent
from world_models.configs import JEPAConfig

# Load pretrained model
cfg = JEPAConfig()
cfg.checkpoint_path = "./checkpoints/jepa_pretrained.pt"

agent = JEPAAgent(cfg)
encoder = agent.encoder

# Extract features
features = encoder(images)  # (B, embed_dim)
```

## Monitoring Training

```python
# Enable wandb
cfg.enable_wandb = True
cfg.wandb_project = "jepa-experiments"

# Enable tensorboard  
cfg.enable_tensorboard = True
cfg.log_dir = "./logs/jepa"
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Representations collapse | Increase predictor depth, adjust mask ratio |
| Slow training | Use gradient accumulation, reduce image size |
| Out of memory | Reduce batch size, use gradient checkpointing |

## Complete Example

```python
import torch
from world_models.configs import JEPAConfig
from world_models.models import JEPAAgent
from world_models.datasets import make_imagefolder

# 1. Configuration
cfg = JEPAConfig()
cfg.dataset = "imagefolder"
cfg.root_path = "./data/imagenet"
cfg.image_folder = "train"
cfg.val_root_path = "./data/imagenet"
cfg.val_image_folder = "val"

cfg.epochs = 100
cfg.batch_size = 64
cfg.embed_dim = 384
cfg.predictor_depth = 6

cfg.lr = 1e-4
cfg.weight_decay = 0.05

cfg.enable_wandb = True
cfg.wandb_project = "jepa-training"

# 2. Create agent
agent = JEPAAgent(cfg)

# 3. Train
agent.train()

# 4. Save feature extractor
torch.save(agent.encoder.state_dict(), "jepa_encoder.pt")
```

## Next Steps

- Explore the Modular RSSM for combining JEPA with latent dynamics
- Check how JEPA representations can be used for downstream tasks
- Review the API documentation for advanced customization
