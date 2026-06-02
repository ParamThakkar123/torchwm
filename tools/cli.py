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
from typer.main import get_command as _typer_get_command
import typer

# Defer heavy optional imports to reduce CLI startup time (gym can be slow).
# These are lazily imported by the helper functions below when a command actually
# needs them.
gym = None
_np = None

# Defer importing the environment catalog until it's needed to avoid pulling in
# package modules at CLI startup. Use `_load_catalog()` below to access it.
_typer_app = typer.Typer(name="torchwm", help="TorchWM command-line tool")


# Some tests (and Click's test runner) expect the top-level CLI object to
# expose attributes like `name` and `main`. Typer.Typer doesn't allow setting
# arbitrary attributes on the object, so create a lightweight proxy that
# forwards attribute access to the underlying Typer instance while exposing
# `name` and a deferred `main` callable compatible with click.testing.CliRunner.


class _TyperProxy:
    def __init__(self, typer_obj, name: str):
        self._typer = typer_obj
        self.name = name

    def __getattr__(self, item):
        # Forward any unknown attribute access to the underlying Typer
        return getattr(self._typer, item)

    def main(self, *args, **kwargs):
        # Resolve the underlying click Command at call time so all subcommands
        # and registrations have been applied.
        return _typer_get_command(self._typer).main(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # Allow the proxy to be invoked like a Typer object (entrypoint
        # compatibility). Forward to the underlying Typer's __call__.
        return self._typer(*args, **kwargs)


app = _TyperProxy(_typer_app, name="torchwm")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("torchwm.cli")


# Lightweight mapping of training entrypoints. Pulled out so other commands can
# reference the list of supported models without importing heavy modules.
TRAINING_MODULES = {
    "iris": "world_models.training.train_iris",
    "planet": "world_models.training.train_planet",
    "jepa": "world_models.training.train_jepa",
    "rssm": "world_models.training.train_rssm",
    "genie": "world_models.training.train_genie",
}


BENCHMARK_AGENT_NAMES = ("diamond", "iris", "dreamerv1", "dreamerv2")


def _parse_benchmark_seeds(value: str) -> List[int]:
    """Parse benchmark seed syntax shared with the benchmark module CLI."""
    if "," in value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if value.isdigit():
        return list(range(int(value)))
    return [int(value)]


def _load_benchmark_runtime():
    """Lazy-load benchmark runtime classes after lightweight validation passes."""
    from world_models.benchmarks.cli import AGENTS
    from world_models.benchmarks.runner import (
        BenchmarkRunner,
        MultiAgentBenchmarkRunner,
    )
    import torch

    return AGENTS, BenchmarkRunner, MultiAgentBenchmarkRunner, torch


def _parse_checkpoint_map(values: List[str] | None) -> dict[str, str]:
    """Parse repeated ``--checkpoint-map agent=path`` values."""
    checkpoints: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Invalid checkpoint map '{value}'. Expected AGENT=PATH, for example iris=checkpoints/iris.pt."
            )
        agent, path = value.split("=", 1)
        agent = agent.strip().lower()
        path = path.strip()
        if not agent or not path:
            raise ValueError(
                f"Invalid checkpoint map '{value}'. Agent and path must both be non-empty."
            )
        checkpoints[agent] = path
    return checkpoints


def _ensure_gym():
    """Lazy-import `gym` (fall back to `gymnasium`) and cache result.

    Returns the imported module or None if import failed.
    """
    try:
        import gym as _gym
    except Exception:
        import gymnasium as _gym
    return _gym


def _ensure_numpy():
    """Lazy-import numpy and cache result. Returns module or None."""
    try:
        import numpy as _np
    except Exception:
        pass
    return _np


def _load_catalog() -> dict:
    """Lazy-load the environment catalog (ENV_BACKENDS).

    Returns a dict mapping backends -> info, or an empty dict on failure.
    """
    try:
        from world_models.catalog import ENV_BACKENDS as _envs

        return _envs
    except Exception:
        logger.debug("Could not import world_models.catalog", exc_info=True)
        return {}


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
models_app = typer.Typer()
app.add_typer(envs_app, name="envs")
app.add_typer(datasets_app, name="datasets")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list() -> None:
    """List supported models and training entrypoints."""
    try:
        print("Training entrypoints:")
        for k, m in sorted(TRAINING_MODULES.items()):
            print(f"- {k}: {m}")

        # Also show model classes/factory names exported by world_models.models
        try:
            import world_models.models as wm_models

            exported = getattr(wm_models, "__all__", None)
            if exported:
                print("\nExported model names:")
                for name in sorted(exported):
                    print(f"- {name}")
        except Exception:
            # Avoid failing the whole command if importing the models package
            # triggers heavier imports; the training entrypoints above are
            # usually sufficient for users wanting to know supported models.
            pass
    except Exception as e:
        logger.exception("Failed to list models: %s", e)
        raise typer.Exit(code=1)


@envs_app.command("list")
def envs_list() -> None:
    """List built-in environments supported by the UI registry."""
    try:
        catalog = _load_catalog()
        if not catalog:
            print(
                "No environment catalog available (world_models.catalog failed to import)."
            )
            raise typer.Exit(code=0)

        for backend, info in catalog.items():
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
    path: Path | None = typer.Argument(None, help="Path to dataset cache"),
) -> None:
    """List datasets in a folder (defaults to TORCHWM_HOME or ~/.torchwm)."""
    home = Path(os.environ.get("TORCHWM_HOME", Path.home() / ".torchwm"))
    root = Path(path) if path is not None else home
    # If the path doesn't exist or contains no entries, report no datasets.
    try:
        has_entries = any(root.iterdir()) if root.exists() else False
    except PermissionError:
        has_entries = False

    if not root.exists() or not has_entries:
        print(f"No datasets found at {root}")
        raise typer.Exit(code=0)

    print(f"Datasets under: {root}")
    for p in sorted(root.iterdir()):
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

    # Import conversion dependencies only when this command is invoked so the
    # rest of the CLI can start in lightweight environments.
    from world_models.datasets.video_datasets import HDF5Dataset, NumPyDataset
    from world_models.utils.utils import save_video

    if _ensure_numpy() is None:
        logger.exception("Missing numpy")
        print("Install the optional dependencies (numpy, h5py, etc.) and retry.")
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
            if getattr(arr, "ndim", None) == 4 and getattr(
                arr, "shape", [None, None, None, None]
            )[1] in (1, 3, 4):
                # convert CHW -> HWC: (T,C,H,W) -> (T,H,W,C')
                try:
                    arr = arr.transpose(0, 2, 3, 1)
                except Exception:
                    # Fallback: try numpy moveaxis if available
                    try:
                        import numpy as _tmpn

                        arr = _tmpn.moveaxis(arr, 1, -1)
                    except Exception:
                        pass
            # convert uint8 [0,255] to float [0,1]
            try:
                dtype_name = getattr(arr, "dtype", None)
                if (
                    dtype_name is not None
                    and str(dtype_name).endswith("uint8")
                    or (hasattr(arr, "max") and float(arr.max()) > 1.0)
                ):
                    # prefer numpy operations if possible
                    try:
                        arrf = (arr.astype("float32") / 255.0).clip(0.0, 1.0)
                    except Exception:
                        arrf = (arr * (1.0 / 255.0)).clip(0.0, 1.0)
                else:
                    try:
                        arrf = arr.astype("float32")
                    except Exception:
                        arrf = arr.astype("float") if hasattr(arr, "astype") else arr
            except Exception:
                arrf = arr

            out_name = out / f"ep_{i:06d}"
            save_video(arrf, str(out_name.parent), out_name.name)
            print(f"Wrote: {out_name}.mp4")
        except Exception as e:
            logger.exception("Failed to convert item %d: %s", i, e)
            print(f"Failed to convert item {i}: {e}")


@app.command("benchmark")
def benchmark(
    agent: str | None = typer.Option(
        None,
        "--agent",
        "-a",
        help="Agent adapter to evaluate (for example: iris, diamond, dreamerv1, dreamerv2).",
    ),
    all_agents: bool = typer.Option(
        False,
        "--all-agents",
        help="Run all registered benchmark adapters on the same environment.",
    ),
    game: str = typer.Option(
        ...,
        "--game",
        "-g",
        help="Gym/ALE environment id to benchmark, such as ALE/Pong-v5.",
    ),
    seeds: str = typer.Option(
        "1",
        "--seeds",
        help="Either N (creates seeds 0..N-1) or a comma-separated seed list.",
    ),
    episodes: int = typer.Option(
        5, "--episodes", "-n", help="Evaluation episodes to run per seed."
    ),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Checkpoint path for a single-agent benchmark.",
    ),
    checkpoint_map: List[str] | None = typer.Option(
        None,
        "--checkpoint-map",
        help="For --all-agents, repeat as AGENT=PATH (for example --checkpoint-map iris=ckpt.pt).",
    ),
    out_dir: Path = typer.Option(
        Path("results/bench"),
        "--out-dir",
        "--out_dir",
        help="Directory where JSON/CSV/Markdown/LaTeX benchmark reports are written.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device passed to adapters. Defaults to CUDA when available, otherwise CPU.",
    ),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help="Optional adapter/model preset to forward to benchmark adapters.",
    ),
    train_epochs: int | None = typer.Option(
        None,
        "--train-epochs",
        help="For --all-agents only: train agents first for this many epochs if no checkpoints are supplied.",
    ),
) -> None:
    """Run TorchWM benchmark evaluations and write report artifacts.

    This is the packaged `torchwm` entrypoint for the benchmark harness that can
    also be invoked with `python -m world_models.benchmarks.cli`. Benchmarking
    trained agents requires checkpoints; `--train-epochs` is available for
    multi-agent smoke runs that intentionally train before evaluating.
    """
    try:
        if not all_agents and not agent:
            print("Either --agent or --all-agents must be specified")
            raise typer.Exit(code=1)

        if all_agents and agent:
            print("--agent and --all-agents cannot be used together")
            raise typer.Exit(code=1)

        if agent:
            agent = agent.strip().lower()
            if agent not in BENCHMARK_AGENT_NAMES:
                print(
                    f"Unknown benchmark agent '{agent}'. Known: {', '.join(sorted(BENCHMARK_AGENT_NAMES))}"
                )
                raise typer.Exit(code=1)
            if checkpoint is None:
                print(
                    "--checkpoint is required when using --agent. Only trained models should be benchmarked."
                )
                raise typer.Exit(code=1)

        seed_values = _parse_benchmark_seeds(seeds)
        env_spec = {"game": game}

        if all_agents:
            checkpoints = _parse_checkpoint_map(checkpoint_map)
            unknown_checkpoints = sorted(
                set(checkpoints).difference(BENCHMARK_AGENT_NAMES)
            )
            if unknown_checkpoints:
                print(
                    f"Unknown benchmark checkpoint agent(s): {', '.join(unknown_checkpoints)}. Known: {', '.join(sorted(BENCHMARK_AGENT_NAMES))}"
                )
                raise typer.Exit(code=1)
            if not checkpoints and train_epochs is None:
                print(
                    "Provide --checkpoint-map AGENT=PATH at least once, or provide --train-epochs for --all-agents."
                )
                raise typer.Exit(code=1)

        AGENTS, BenchmarkRunner, MultiAgentBenchmarkRunner, torch = (
            _load_benchmark_runtime()
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        extra_kwargs = {"device": device, "preset": preset}

        if all_agents:
            runner = MultiAgentBenchmarkRunner(
                adapters=list(AGENTS.values()), out_dir=str(out_dir)
            )
            runner.run_all(
                env_spec=env_spec,
                seeds=seed_values,
                num_episodes=episodes,
                checkpoints=checkpoints or None,
                extra_kwargs=extra_kwargs,
                train_epochs=train_epochs,
            )
            print(f"Multi-agent benchmark finished. Results written to: {out_dir}")
            raise typer.Exit(code=0)

        assert agent is not None
        assert checkpoint is not None
        runner = BenchmarkRunner(adapter_cls=AGENTS[agent], out_dir=str(out_dir))
        runner.run(
            env_spec=env_spec,
            seeds=seed_values,
            num_episodes=episodes,
            checkpoint=str(checkpoint),
            extra_kwargs=extra_kwargs,
        )
        print(f"Benchmark finished. Results written to: {out_dir}")
    except ValueError as e:
        print(str(e))
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        print("Benchmark interrupted by user")
        raise typer.Exit(code=1)


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
    # Lazy-import optional dependencies (gym/gymnasium and numpy)
    _gym = _ensure_gym()
    _n = _ensure_numpy()
    if _gym is None or _n is None:
        logger.exception("Missing gym or numpy")
        print("Please install gym (or gymnasium) and numpy to use collect")
        raise typer.Exit(code=1)

    print(f"Creating environment: {env}")
    try:
        # Try world_models envs first
        try:
            from world_models.envs import make_env

            env_obj = make_env(env)
        except Exception:
            env_obj = _gym.make(env)
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
        obs_list.append(_n.asarray(obs))
        act_list.append(_n.asarray(action))
        rew_list.append(float(reward))
        done_list.append(bool(done))
        if done:
            obs = env_obj.reset()
            if isinstance(obs, tuple) and len(obs) >= 1:
                obs = obs[0]
        else:
            obs = next_obs

    print(f"Collected {len(obs_list)} steps; saving to {out}")
    _n.savez_compressed(
        str(out),
        observations=_n.stack(obs_list),
        actions=_n.stack(act_list),
        rewards=_n.array(rew_list),
        dones=_n.array(done_list),
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
    inproc: bool = typer.Option(
        False, help="Run training in-process instead of spawning subprocess"
    ),
) -> None:
    """Launch an existing training entrypoint as a subprocess.

    This command spawns `python -m world_models.training.<script>` for the
    requested model. Extra args are forwarded unchanged.
    """
    # Use the shared training module mapping defined near the top of this file.
    mapping = TRAINING_MODULES

    key = model.strip().lower()
    if key not in mapping:
        print(f"Unknown model '{model}'. Known: {', '.join(mapping.keys())}")
        raise typer.Exit(code=1)

    module = mapping[key]
    if inproc:
        # Try to import the module and call a `main` entrypoint directly.
        try:
            import importlib

            mod = importlib.import_module(module)
            main_fn = getattr(mod, "main", None)
            if callable(main_fn):
                print(f"Running in-process: {module}.main()")
                # If main expects args, we try to pass nothing and let it parse sys.argv
                try:
                    main_fn()
                except TypeError:
                    # Some mains accept args; attempt to pass extra_args if provided
                    if extra_args:
                        main_fn(extra_args)
                    else:
                        main_fn()
                raise typer.Exit(code=0)
            else:
                print(
                    f"Module {module} has no callable main(); falling back to subprocess"
                )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            raise typer.Exit(code=1)
        except Exception as e:
            logger.exception("In-process training failed: %s", e)
            print("Falling back to subprocess execution")

    # Spawn subprocess as fallback / default
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


if __name__ == "__main__":
    app()
