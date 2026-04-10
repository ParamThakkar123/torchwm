"""Snapshot persistence and CLI for the simulator.

Provides utilities to save/load snapshots to disk and a simple CLI for
generating episodes with various export formats.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Union

from torchwm.sim.envs.basic_env import BasicEnv
from torchwm.sim.exporters.hdf5 import HDF5Exporter
from torchwm.sim.exporters.image_json import ImageJSONExporter


def save_snapshot(env: BasicEnv, path: str) -> None:
    """Save environment snapshot to disk."""
    snapshot = env.snapshot()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(snapshot, f)
    print(f"Saved snapshot to {path}")


def load_snapshot(env: BasicEnv, path: str) -> None:
    """Load environment snapshot from disk."""
    with open(path, "rb") as f:
        snapshot = pickle.load(f)
    env.restore(snapshot)
    print(f"Loaded snapshot from {path}")


def run_generate(
    episodes: int,
    steps: int,
    out_path: str,
    mode: str = "hdf5",
    seed_start: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Run episode generation with specified export mode."""
    if config is None:
        config = make_default_config()

    env = BasicEnv(config)

    exporter: Union[HDF5Exporter, ImageJSONExporter]
    if mode == "hdf5":
        exporter = HDF5Exporter(out_path, mode="w")
    elif mode == "image_json":
        exporter = ImageJSONExporter(out_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with exporter:
        for ep in range(episodes):
            seed = seed_start + ep
            obs, info = env.reset(seed=seed)
            frames: List[Any] = []
            actions: List[List[float]] = []
            rewards: List[float] = []
            dones: List[bool] = []

            for t in range(steps):
                action: List[float] = []
                obs, reward, done, info = env.step(action)
                frames.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                if done:
                    break

            metadata = {"seed": seed, "steps": len(frames)}
            exporter.add_episode(
                frames=frames,
                actions=actions,
                rewards=rewards,
                dones=dones,
                metadata=metadata,
            )
            print(f"Episode {ep} done (seed={seed})")

    env.close()
    print(f"Generated {episodes} episodes -> {out_path}")


def make_default_config() -> Dict[str, Any]:
    return {
        "physics": {
            "timestep": 1.0 / 60.0,
            "substeps": 1,
            "num_solver_iterations": 50,
            "gravity_z": -9.81,
        },
        "generator": {
            "objects": [
                {
                    "shape": {"type": "box", "size": [0.5, 0.5, 0.5]},
                    "position": [0.0, 0.0, 1.0],
                    "mass": 1.0,
                },
            ]
        },
        "camera": {
            "width": 64,
            "height": 64,
            "fov": 60.0,
            "position": [1.0, 1.0, 1.0],
            "target": [0.0, 0.0, 0.0],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="TorchWM Simulator CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate episodes")
    gen_parser.add_argument("--episodes", type=int, default=10)
    gen_parser.add_argument("--steps", type=int, default=100)
    gen_parser.add_argument("--out", type=str, required=True)
    gen_parser.add_argument(
        "--mode", type=str, default="hdf5", choices=["hdf5", "image_json"]
    )
    gen_parser.add_argument("--seed-start", type=int, default=0)

    # snapshot subcommand
    snap_parser = subparsers.add_parser("snapshot", help="Save/load snapshot")
    snap_parser.add_argument("action", choices=["save", "load"])
    snap_parser.add_argument("path", help="Snapshot file path")
    snap_parser.add_argument(
        "--config", type=str, help="Config JSON file (for load action)"
    )

    args = parser.parse_args()

    if args.command == "generate":
        run_generate(
            episodes=args.episodes,
            steps=args.steps,
            out_path=args.out,
            mode=args.mode,
            seed_start=args.seed_start,
        )
    elif args.command == "snapshot":
        config = make_default_config()
        if args.action == "save":
            env = BasicEnv(config)
            env.reset(seed=0)
            save_snapshot(env, args.path)
            env.close()
        elif args.action == "load":
            if args.config:
                with open(args.config) as f:
                    config = json.load(f)
            env = BasicEnv(config)
            load_snapshot(env, args.path)
            # run a few steps to verify
            for _ in range(5):
                env.step({})
            env.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
