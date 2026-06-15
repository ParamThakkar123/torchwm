from dataclasses import dataclass, replace
from typing import Any

from world_models.configs.serialization import SerializableConfigMixin

_SNAKE_TO_UPPER = {
    "dataset": "DATASET",
    "batch": "BATCH",
    "epochs": "EPOCHS",
    "lr": "LR",
    "img_size": "IMG_SIZE",
    "channels": "CHANNELS",
    "patch_size": "PATCH",
    "width": "WIDTH",
    "depth": "DEPTH",
    "heads": "HEADS",
    "drop": "DROP",
    "beta_start": "BETA_START",
    "beta_end": "BETA_END",
    "timesteps": "TIMESTEPS",
    "ema": "EMA",
    "ema_decay": "EMA_DECAY",
    "workdir": "WORKDIR",
    "root_path": "ROOT_PATH",
}


@dataclass
class DiTConfig(SerializableConfigMixin):
    """Default configuration values for Diffusion Transformer (DiT) training.

    The fields define dataset selection, model architecture, diffusion schedule,
    optimization hyperparameters, and output paths used by the built-in
    training entrypoints.

    Field names use UPPER_CASE for backward compatibility with the original DiT
    codebase. Snake-case aliases are accepted via ``__getattr__`` and
    ``get_dit_config()``.
    """

    DATASET: str = "CIFAR10"
    BATCH: int = 128
    EPOCHS: int = 3
    LR: float = 2e-4
    IMG_SIZE: int = 32
    CHANNELS: int = 3
    PATCH: int = 4
    WIDTH: int = 384
    DEPTH: int = 6
    HEADS: int = 6
    DROP: float = 0.1
    BETA_START: float = 1e-4
    BETA_END: float = 0.02
    TIMESTEPS: int = 1000
    EMA: bool = True
    EMA_DECAY: float = 0.999
    WORKDIR: str = "./dit_demo"
    ROOT_PATH: str = "./data"

    def __getattr__(self, name: str) -> Any:
        upper = _SNAKE_TO_UPPER.get(name)
        if upper is not None:
            return getattr(self, upper)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        upper = _SNAKE_TO_UPPER.get(name, name)
        super().__setattr__(upper, value)


def get_dit_config(**overrides: Any) -> DiTConfig:
    """
    Returns a DiTConfig instance with default values overridden by the provided keyword arguments.

    Both UPPER_CASE and snake_case override keys are accepted.

    Example usage:
        cfg = get_dit_config(BATCH=64, EPOCHS=10, LR=1e-3)
        cfg = get_dit_config(batch=64, epochs=10, lr=1e-3)
    """
    translated = {}
    for key, value in overrides.items():
        translated[_SNAKE_TO_UPPER.get(key, key)] = value
    return replace(DiTConfig(), **translated)
