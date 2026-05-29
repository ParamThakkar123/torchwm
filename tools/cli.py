"""Command-line interface entrypoint for torchwm.

Minimal implementation with Typer providing `torchwm` console group and
subcommands: `envs list`, `datasets list`, and `serve` to run the existing
FastAPI UI. This initial patch avoids adding heavy new logic: it reuses the
server in `world_models.ui.server` and dataset listing helpers.

Run: python -m tools.cli <command>
"""

from __future__ import annotations

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List

try:
    import typer
except Exception:  # pragma: no cover - runtime guard
    print("Please install 'typer' to use the CLI: pip install typer[all]")
    raise

app = typer.Typer(help="TorchWM command-line tool")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("torchwm.cli")


@app.command("version")
def version() -> None:
    """Show package version."""
    try:
        import torchwm as pkg

        print(pkg.__version__)
    except Exception:
        print("torchwm (unknown version)")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


envs_app = typer.Typer()
datasets_app = typer.Typer()
app.add_typer(envs_app, name="envs")
app.add_typer(datasets_app, name="datasets")


@envs_app.command("list")
def envs_list() -> None:
    """List built-in environments supported by the UI registry."""
    try:
        from world_models.ui.server import ENV_BACKENDS

        for backend, info in ENV_BACKENDS.items():
            print(f"{backend}: {info.get('label', '')}")
            items = info.get("environments", [])
            if items:
                for env in items[:20]:
                    print(f"  - {env}")
                if len(items) > 20:
                    print(f"  ... and {len(items) - 20} more")
    except Exception as e:
        logger.exception("Failed to list environments: %s", e)
        raise typer.Exit(code=1)


@datasets_app.command("list")
def datasets_list(
    path: Path = typer.Option(None, help="Path to dataset cache"),
) -> None:
    """List datasets in a folder (defaults to TORCHWM_HOME or ~/.torchwm)."""
    home = Path(os.environ.get("TORCHWM_HOME", Path.home() / ".torchwm"))
    root = Path(path) if path is not None else home
    if not root.exists():
        print(f"No datasets found at {root}")
        raise typer.Exit(code=0)
    print(f"Datasets under: {root}")
    for p in sorted(root.iterdir()):
        if p.is_dir():
            print(f"- {p.name}")
        else:
            print(f"- {p.name}")


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Run the FastAPI UI server (thin wrapper around uvicorn).

    This command delegates to uvicorn to avoid duplicating the ASGI startup.
    """
    # Ensure importable package path when running from repo root
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import uvicorn  # type: ignore
    except Exception:
        print("Please install uvicorn to serve the UI: pip install 'uvicorn[standard]'")
        raise typer.Exit(code=1)

    module = "world_models.ui.server:app"
    url = f"http://{host}:{port}"
    print(f"Starting UI at {url} (module={module})")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run("world_models.ui.server:app", host=host, port=port, log_level="info")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
