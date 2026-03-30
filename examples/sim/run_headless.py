"""Headless generation script example.

Generates N episodes using BasicEnv, records frames and exports to HDF5.

Usage:
    python -m examples.sim.run_headless --episodes 10 --steps 100 --out data.h5
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List
import numpy as np

from torchwm.sim.envs.basic_env import BasicEnv
from torchwm.sim.exporters.hdf5 import HDF5Exporter
from torchwm.sim.exporters.image_json import ImageJSONExporter


def make_config() -> Dict[str, Any]:
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


def run(episodes: int, steps: int, out_path: str, mode: str = "hdf5") -> None:
    cfg = make_config()
    env = BasicEnv(cfg)
    if mode == "hdf5":
        exporter = HDF5Exporter(out_path, mode="a")
    elif mode == "image_json":
        exporter = ImageJSONExporter(out_path)
    else:
        raise ValueError(f"Unknown export mode: {mode}")

    with exporter:
        for ep in range(episodes):
            seed = int(ep)
            obs, info = env.reset(seed=seed)
            frames: List[Any] = []
            actions: List[Any] = []
            rewards: List[float] = []
            dones: List[bool] = []

            for t in range(steps):
                # Action space: none for this simple env. Record placeholder.
                action = []
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

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out", dest="out", type=str, default="data/sim_data.h5")
    parser.add_argument(
        "--mode", dest="mode", type=str, default="hdf5", choices=["hdf5", "image_json"]
    )
    args = parser.parse_args()
    run(args.episodes, args.steps, args.out, mode=args.mode)


if __name__ == "__main__":
    main()
