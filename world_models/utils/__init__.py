"""
Utils sub-module - Utility functions and classes.

Exported Components:
    - Logger: Experiment logger for scalars and GIF rollouts
    - FreezeParameters: Context manager to disable gradients
    - get_parameters: Extract parameters from modules
    - compute_return: Compute returns for value estimation
    - preprocess_obs: Preprocess observations for Dreamer
"""

__all__ = [
    "Logger",
    "FreezeParameters",
    "get_parameters",
    "compute_return",
    "preprocess_obs",
]


def __getattr__(name):
    if name in ("Logger", "FreezeParameters", "get_parameters", "compute_return"):
        from .dreamer_utils import (
            Logger,
            FreezeParameters,
            get_parameters,
            compute_return,
        )

        return locals()[name]
    if name == "preprocess_obs":
        from world_models.models.dreamer import preprocess_obs

        return preprocess_obs

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
