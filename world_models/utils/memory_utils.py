import torch
import torch.nn as nn
from typing import Optional


def apply_gradient_checkpointing(model: nn.Module, checkpoint_ratio: float = 0.5):
    """Apply gradient checkpointing to reduce memory usage during training."""
    # Some models expose gradient_checkpointing_enable as a callable while
    # others may have an attribute with the same name that isn't callable.
    # Use getattr and only call when it's actually callable to avoid mypy
    # complaining about "Tensor" not callable at type-check time.
    gc_enable = getattr(model, "gradient_checkpointing_enable", None)
    if callable(gc_enable):
        gc_enable()
    else:
        # For custom modules, apply selective checkpointing
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                # Wrap the original forward in a callable that uses
                # torch.utils.checkpoint.checkpoint. We capture the original
                # method to avoid recursive lookup and assign a plain
                # function to the instance attribute (allowed at runtime).
                from typing import Callable, Any

                orig_forward: Callable[..., Any] = module.forward  # capture

                def _checkpointed_forward(*args: Any, **kwargs: Any) -> Any:
                    # Torch's checkpoint API is present at runtime but some stubs
                    # do not expose it. Ignore attribute errors from type-checker
                    # here while preserving runtime behavior.
                    return torch.utils.checkpoint.checkpoint(  # type: ignore[attr-defined]
                        orig_forward, *args, **kwargs, use_reentrant=False
                    )

                # Assigning a function to an instance method is a runtime pattern
                # used to wrap behavior; mypy may complain about assigning to
                # a method attribute, so silence that specific check.
                setattr(module, "forward", _checkpointed_forward)  # type: ignore[assignment]
            elif hasattr(module, "checkpoint_forward"):
                # Create a wrapper that calls checkpoint at runtime. Do not
                # call checkpoint here (that would execute the function and
                # assign a Tensor to `forward`). Some torch stubs do not
                # expose `utils.checkpoint`, so silence attribute checks.
                from typing import Any

                def _checkpointed_forward2(*args: Any, **kwargs: Any) -> Any:
                    # Use a targeted ignore for the missing `checkpoint` attr in
                    # some torch stubs while preserving runtime behaviour.
                    return torch.utils.checkpoint.checkpoint(  # type: ignore[attr-defined]
                        module.checkpoint_forward, *args, **kwargs, use_reentrant=False
                    )

                setattr(module, "forward", _checkpointed_forward2)  # type: ignore[assignment]


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
