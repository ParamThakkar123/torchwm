"""Minimal setup shim for legacy pip editable-install compatibility.

All canonical metadata lives in ``pyproject.toml``.
"""

from setuptools import setup, find_namespace_packages

setup(packages=find_namespace_packages(include=["world_models*", "torchwm*"]))
