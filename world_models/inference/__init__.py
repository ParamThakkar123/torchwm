"""
Inference sub-module - Operators for running trained world models.

This package provides inference operators that abstract the common interface
for running different world model types (Dreamer, JEPA, IRIS, PlaNet).

Usage:
    from world_models.inference import get_operator
    op = get_operator('dreamer', image_size=64, action_dim=6)
"""

from .operators import (
    OperatorABC,
    DreamerOperator,
    JEPAOperator,
    IrisOperator,
    PlaNetOperator,
    get_operator,
)

__all__ = [
    "OperatorABC",
    "DreamerOperator",
    "JEPAOperator",
    "IrisOperator",
    "PlaNetOperator",
    "get_operator",
]
