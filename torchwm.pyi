"""Typing stub for the friendly ``torchwm`` public package."""

from typing import Any

__version__: str
api: Any


def __getattr__(name: str) -> Any: ...
