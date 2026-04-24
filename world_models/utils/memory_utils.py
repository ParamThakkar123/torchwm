import torch
import torch.nn as nn
from functools import partial
from typing import Optional, Callable


def apply_gradient_checkpointing(model: nn.Module, checkpoint_ratio: float = 0.5):
    """Apply gradient checkpointing to reduce memory usage during training."""
    if hasattr(model, "gradient_checkpointing_enable"):
        # For transformers
        model.gradient_checkpointing_enable()
    else:
        # For custom modules, apply selective checkpointing
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                module.forward = torch.utils.checkpoint.checkpoint(
                    module.forward, use_reentrant=False
                )
            elif hasattr(module, "checkpoint_forward"):
                module.forward = torch.utils.checkpoint.checkpoint(
                    module.checkpoint_forward, use_reentrant=False
                )


def enable_mixed_precision(
    model: nn.Module, scaler: Optional[torch.cuda.amp.GradScaler] = None
):
    """Enable mixed precision training."""
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler()
    return scaler


def optimize_memory_efficient_ops():
    """Set PyTorch for memory-efficient operations."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
