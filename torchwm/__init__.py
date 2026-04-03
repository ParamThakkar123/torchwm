"""Top-level package marker for editable installs and static checkers.

This package exists so tools like mypy can map files under the repository
to the `torchwm` package name (the project uses a flat layout with
`world_models/` as the real package). It intentionally re-exports the
public modules where appropriate.
"""

# Keep this file minimal — importing subpackages lazily avoids heavy
# import-time dependencies during type checking.
__all__ = []
