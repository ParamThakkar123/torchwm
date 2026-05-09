"""
Benchmarks sub-module - Benchmark runners and adapters for world models.

This package provides tools for running standardized evaluations of world
models (Dreamer, IRIS, DIAMOND) across multiple seeds and computing aggregate
metrics.

Usage:
    from world_models.benchmarks import BenchmarkRunner, DiamondAdapter
    runner = BenchmarkRunner(adapter_cls=DiamondAdapter, ...)
"""

from .runner import BenchmarkRunner, MultiAgentBenchmarkRunner
from .adapters import (
    BaseAdapter,
    DreamerAdapter,
    IRISAdapter,
    DiamondAdapter,
)
from .metrics import compute_aggregate_metrics, bootstrap_ci, iqm_of_array

__all__ = [
    "BenchmarkRunner",
    "MultiAgentBenchmarkRunner",
    "BaseAdapter",
    "DreamerAdapter",
    "IRISAdapter",
    "DiamondAdapter",
    "compute_aggregate_metrics",
    "bootstrap_ci",
    "iqm_of_array",
]
