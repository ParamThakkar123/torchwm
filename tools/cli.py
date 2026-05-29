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


@datasets_app.command("convert")
def datasets_convert(
    src: Path = typer.Argument(..., help="Source dataset file (h5/npz/npy or folder)"),
    dest_format: str = typer.Option("video", help="Destination format: video"),
    out_dir: Path = typer.Option(None, help="Output directory for converted data"),
) -> None:
    """Convert simple dataset files into another format.

    Current supported conversion: hdf5(.h5) -> mp4 files (one per episode) when
    `--dest-format video` is used.
    """
    out = Path(out_dir) if out_dir is not None else Path.cwd() / "converted_datasets"
    out.mkdir(parents=True, exist_ok=True)

    if dest_format != "video":
        print("Only 'video' dest_format supported in this initial implementation.")
        raise typer.Exit(code=1)

    # Lazy import heavy modules only when needed
    try:
        from world_models.datasets.video_datasets import HDF5Dataset, NumPyDataset
        from world_models.utils.utils import save_video
    except Exception as e:
        logger.exception("Missing dataset conversion dependencies: %s", e)
        print("Install the optional dependencies (h5py, etc.) and retry.")
        raise typer.Exit(code=1)

    srcp = Path(src)
    if not srcp.exists():
        print(f"Source not found: {src}")
        raise typer.Exit(code=1)

    if srcp.suffix in {".h5", ".hdf5"}:
        ds = HDF5Dataset(str(srcp), num_frames=16, image_size=64)
    elif srcp.suffix in {".npz", ".npy"}:
        ds = NumPyDataset(str(srcp), num_frames=16, image_size=64)
    else:
        print("Unsupported source format for conversion. Provide .h5 or .npz/.npy")
        raise typer.Exit(code=1)

    total = len(ds)
    print(f"Converting {total} items from {src} -> {out} ...")
    for i in range(total):
        try:
            v = ds[i]
            # v may be torch.Tensor or numpy array
            if hasattr(v, "numpy"):
                arr = v.numpy()
            else:
                arr = v
            # arr expected shape: (T,H,W,C) or (T,C,H,W)
            if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
                # convert CHW -> HWC: (T,C,H,W) -> (T,H,W,C)
                arr = arr.transpose(0, 2, 3, 1)
            # convert uint8 [0,255] to float [0,1]
            if arr.dtype == "uint8" or arr.max() > 1.0:
                arrf = (arr.astype("float32") / 255.0).clip(0.0, 1.0)
            else:
                arrf = arr.astype("float32")

            out_name = out / f"ep_{i:06d}"
            save_video(arrf, str(out_name.parent), out_name.name)
            print(f"Wrote: {out_name}.mp4")
        except Exception as e:
            logger.exception("Failed to convert item %d: %s", i, e)
            print(f"Failed to convert item {i}: {e}")


@app.command("collect")
def collect(
    env: str = typer.Option(..., help="Environment id (Gym/Atari)"),
    steps: int = typer.Option(1000, help="Number of environment steps to collect"),
    out: Path = typer.Option(Path("./collected.npz"), help="Output file path (.npz)"),
    random_policy: bool = typer.Option(True, help="Use random actions"),
) -> None:
    """Collect interactions from an environment and save as a simple .npz file.

    The saved file contains keys: observations, actions, rewards, dones.
    """
    try:
        import gym
        import numpy as _np
    except Exception as e:
        logger.exception("Missing gym or numpy: %s", e)
        print("Please install gym and numpy to use collect")
        raise typer.Exit(code=1)

    print(f"Creating environment: {env}")
    try:
        # Try world_models envs first
        try:
            from world_models.envs import make_env

            env_obj = make_env(env)
        except Exception:
            env_obj = gym.make(env)
    except Exception as e:
        logger.exception("Failed to create env %s: %s", env, e)
        print(f"Failed to create environment: {e}")
        raise typer.Exit(code=1)

    obs_list = []
    act_list = []
    rew_list = []
    done_list = []

    obs = env_obj.reset()
    # gym vs gymnasium differences: unpack tuple if needed
    if isinstance(obs, tuple) and len(obs) >= 1:
        obs = obs[0]

    for step in range(steps):
        if random_policy:
            action = env_obj.action_space.sample()
        else:
            # default to random if no policy provided
            action = env_obj.action_space.sample()
        res = env_obj.step(action)
        if isinstance(res, tuple) and len(res) == 5:
            next_obs, reward, terminated, truncated, info = res
            done = terminated or truncated
        else:
            next_obs, reward, done, info = res
        obs_list.append(_np.asarray(obs))
        act_list.append(_np.asarray(action))
        rew_list.append(float(reward))
        done_list.append(bool(done))
        if done:
            obs = env_obj.reset()
            if isinstance(obs, tuple) and len(obs) >= 1:
                obs = obs[0]
        else:
            obs = next_obs

    print(f"Collected {len(obs_list)} steps; saving to {out}")
    _np.savez_compressed(
        str(out),
        observations=_np.stack(obs_list),
        actions=_np.stack(act_list),
        rewards=_np.array(rew_list),
        dones=_np.array(done_list),
    )
    print("Saved.")


@app.command("train")
def train(
    model: str = typer.Argument(
        ..., help="Model/training script to run (iris/planet/jepa/rssm/genie)"
    ),
    extra_args: List[str] = typer.Argument(
        None, help="Extra args passed to training script"
    ),
) -> None:
    """Launch an existing training entrypoint as a subprocess.

    This command spawns `python -m world_models.training.<script>` for the
    requested model. Extra args are forwarded unchanged.
    """
    mapping = {
        "iris": "world_models.training.train_iris",
        "planet": "world_models.training.train_planet",
        "jepa": "world_models.training.train_jepa",
        "rssm": "world_models.training.train_rssm",
        "genie": "world_models.training.train_genie",
    }

    key = model.strip().lower()
    if key not in mapping:
        print(f"Unknown model '{model}'. Known: {', '.join(mapping.keys())}")
        raise typer.Exit(code=1)

    module = mapping[key]
    cmd = [sys.executable, "-m", module]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=False)
        raise typer.Exit(code=proc.returncode)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        raise typer.Exit(code=1)


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
