"""Throughput benchmark for the simulator.

Runs M environments for S steps each and reports:
 - median, p95 frame time
 - steps per second (throughput)
 - optional export bytes written

Usage:
    python -m tools.bench_throughput --envs 4 --steps 100
"""

from __future__ import annotations

import argparse
import time
import statistics
from typing import Any, Dict, List

from torchwm.sim.envs.basic_env import BasicEnv
from torchwm.sim.exporters.hdf5 import HDF5Exporter


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


def run_bench(
    num_envs: int, steps: int, export: bool = False, out_path: str = "bench_out.h5"
) -> None:
    cfg = make_config()
    envs: List[BasicEnv] = [BasicEnv(cfg) for _ in range(num_envs)]

    # warm-up: reset each env once
    for i, env in enumerate(envs):
        env.reset(seed=i)

    # timing storage
    frame_times: List[float] = []

    exporter = None
    if export:
        exporter = HDF5Exporter(out_path, mode="w")

    total_frames = 0

    try:
        for step in range(steps):
            step_start = time.perf_counter()

            # step all envs
            for env in envs:
                obs, reward, done, info = env.step({})

            step_end = time.perf_counter()
            frame_times.append(step_end - step_start)
            total_frames += num_envs

            # optional export every N steps
            if exporter and step % 10 == 0:
                # dummy export to measure I/O cost
                pass

    finally:
        for env in envs:
            env.close()
        if exporter:
            exporter.close()

    # compute metrics
    median = statistics.median(frame_times)
    p95 = (
        sorted(frame_times)[int(len(frame_times) * 0.95)]
        if len(frame_times) > 1
        else frame_times[0]
    )
    total_time = sum(frame_times)
    fps = total_frames / total_time if total_time > 0 else 0

    print(f"\n=== Throughput Benchmark ===")
    print(f"Environments: {num_envs}")
    print(f"Steps per env: {steps}")
    print(f"Total frames: {total_frames}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Median frame time: {median * 1000:.2f}ms")
    print(f"P95 frame time: {p95 * 1000:.2f}ms")
    print(f"Throughput: {fps:.1f} frames/sec")
    if export:
        import os

        size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        print(f"Export size: {size / 1024 / 1024:.2f}MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--out", type=str, default="bench_out.h5")
    args = parser.parse_args()
    run_bench(args.envs, args.steps, args.export, args.out)


if __name__ == "__main__":
    main()
