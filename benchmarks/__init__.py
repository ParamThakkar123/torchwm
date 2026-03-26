"""
IRIS Benchmark Suite

This module provides tools for running IRIS on the Atari 100k benchmark.

Usage:
    python -m benchmarks.atari_100k --device cuda
"""

from .atari_100k import (
    ATARI_100K_GAMES,
    run_atari_100k,
    run_single_game,
    compute_human_normalized_score,
)

__all__ = [
    "ATARI_100K_GAMES",
    "run_atari_100k",
    "run_single_game",
    "compute_human_normalized_score",
]
