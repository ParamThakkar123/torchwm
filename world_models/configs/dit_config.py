from dataclasses import dataclass, replace


@dataclass
class DiTConfig:
    """Default configuration values for Diffusion Transformer (DiT) training.

    The fields define dataset selection, model architecture, diffusion schedule,
    optimization hyperparameters, and output paths used by the built-in
    training entrypoints.
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


def get_dit_config(**overrides):
    """
    Returns a DiTConfig instance with default values overridden by the provided keyword arguments.

    Example usage:
        cfg = get_dit_config(BATCH=64, EPOCHS=10, LR=1e-3)
    """
    return replace(DiTConfig(), **overrides)
