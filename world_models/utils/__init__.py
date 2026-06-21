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
    "ProjectionResult",
    "plot_projection",
    "project_latent_trajectories",
    "project_representation_embeddings",
    "MetricsLogger",
    "assert_finite",
    "assert_finite_values",
    "collect_system_stats",
    "get_package_logger",
    "setup_logging",
]


from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("Logger", "FreezeParameters", "get_parameters", "compute_return"):
        from .dreamer_utils import (
            Logger,
            FreezeParameters,
            get_parameters,
            compute_return,
        )

        return locals()[name]
    if name in (
        "MetricsLogger",
        "assert_finite",
        "assert_finite_values",
        "collect_system_stats",
        "get_package_logger",
        "setup_logging",
    ):
        from .logging_utils import (
            MetricsLogger,
            assert_finite,
            assert_finite_values,
            collect_system_stats,
            get_package_logger,
            setup_logging,
        )

        return locals()[name]
    if name in (
        "ProjectionResult",
        "plot_projection",
        "project_latent_trajectories",
        "project_representation_embeddings",
    ):
        from .visualization import (
            ProjectionResult,
            plot_projection,
            project_latent_trajectories,
            project_representation_embeddings,
        )

        return locals()[name]
    if name == "preprocess_obs":
        from world_models.models.dreamer import preprocess_obs

        return preprocess_obs

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
