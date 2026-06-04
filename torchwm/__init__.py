"""Friendly top-level package for TorchWM.

``torchwm`` mirrors the public ``world_models`` API so users can choose the
package name they installed::

    import torchwm

    agent = torchwm.create_model("dreamer", env="walker-walk")
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import world_models as _world_models

__version__ = _world_models.__version__
__all__ = list(_world_models.__all__)


def __getattr__(name: str) -> Any:
    try:
        value = getattr(_world_models, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# Keep module tools that introspect subpackages working after the alias import.
world_models = import_module("world_models")
