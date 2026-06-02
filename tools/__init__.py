"""Tools package for torchwm.

This package re-exports the commonly used CLI and tooling helpers so callers
can import them directly from ``tools`` (for example:

    from tools import run
    from tools import check_docs_main

The original implementations live in the submodules to keep code organized.
"""

from .cli import (
    app as cli_app,
    main as cli_main,
    version,
    envs_list,
    datasets_list,
    datasets_convert,
    collect,
    benchmark,
    train,
)

from .check_docs_render import (
    main as check_docs_main,
    check_page,
    file_url,
    OUT_DIR,
)

__all__ = [
    "cli_app",
    "cli_main",
    "version",
    "envs_list",
    "datasets_list",
    "datasets_convert",
    "collect",
    "benchmark",
    "train",
    "check_docs_main",
    "check_page",
    "file_url",
    "OUT_DIR",
]
