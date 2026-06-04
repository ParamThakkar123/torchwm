"""Typing stub for the friendly ``torchwm`` package alias.

The runtime package mirrors the public ``world_models`` API; this stub exposes
those names to static type checkers.
"""

from world_models import *  # re-export for type checking

__all__ = [name for name in globals().keys() if not name.startswith("_")]
