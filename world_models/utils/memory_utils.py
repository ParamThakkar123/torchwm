import torch
import torch.nn as nn
from typing import Optional


def apply_gradient_checkpointing(model: nn.Module, checkpoint_ratio: float = 0.5):
    """Apply gradient checkpointing to reduce memory usage during training."""
    if hasattr(model, "gradient_checkpointing_enable"):
        # For transformers
        model.gradient_checkpointing_enable()
    else:
        # For custom modules, apply selective checkpointing
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                # Wrap the original forward in a callable that uses
                # torch.utils.checkpoint.checkpoint. We capture the original
                # method to avoid recursive lookup and assign a plain
                # function to the instance attribute (allowed at runtime).
                orig_forward = module.forward

                def _checkpointed_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(
                        orig_forward, *args, **kwargs, use_reentrant=False
                    )

                setattr(module, "forward", _checkpointed_forward)
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
