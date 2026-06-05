"""Friendly top-level package for TorchWM.

``torchwm`` is the recommended public namespace for users::

    import torchwm

    agent = torchwm.create_model("dreamer", env="walker-walk")
"""

from __future__ import annotations

from importlib import import_module
import sys
from typing import Any

_world_models = import_module("world_models")
api = import_module("world_models.api")
sys.modules[f"{__name__}.api"] = api

__version__ = _world_models.__version__
__all__ = [*list(_world_models.__all__), "api"]


def __getattr__(name: str) -> Any:
    try:
        value = getattr(_world_models, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
