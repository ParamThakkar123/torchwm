"""Typing-only module to make the historical top-level name
``torchwm`` point to the current package ``world_models`` for static
type checkers (mypy). This is a stub only — it does not create a
runtime package directory and therefore doesn't affect imports at
runtime.
"""

from world_models import *  # re-export for type checking

__all__ = [name for name in globals().keys() if not name.startswith("_")]
