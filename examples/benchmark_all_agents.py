"""Example script to benchmark all world models on the same environment.

This demonstrates the new --all-agents feature that runs benchmarks for all
available world model agents (Diamond, IRIS, DreamerV1, DreamerV2) on the
same environment.

Usage:
    python examples/benchmark_all_agents.py --game ALE/Pong-v5 --seeds 2 --episodes 3
"""

import argparse
import os
import sys

# Add the world_models package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from world_models.benchmarks.runner import MultiAgentBenchmarkRunner
from world_models.benchmarks import adapters


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all world models on the same environment"
    )
    parser.add_argument("--game", type=str, required=True, help="Game environment name")
    parser.add_argument("--seeds", type=int, default=2, help="Number of seeds to run")
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes per seed"
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/bench_all", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")

    args = parser.parse_args()

    # List of all available adapters
    all_adapters = [
        adapters.DiamondAdapter,
        adapters.IRISAdapter,
        adapters.DreamerV1Adapter,
        adapters.DreamerV2Adapter,
    ]

    print(f"Benchmarking all agents on {args.game}")
    print(f"Agents: {[cls.__name__ for cls in all_adapters]}")
    print(f"Seeds: {args.seeds}, Episodes per seed: {args.episodes}")
    print(f"Output directory: {args.out_dir}")

    runner = MultiAgentBenchmarkRunner(adapters=all_adapters, out_dir=args.out_dir)

    results = runner.run_all(
        env_spec={"game": args.game},
        seeds=list(range(args.seeds)),
        num_episodes=args.episodes,
        extra_kwargs={"device": args.device},
    )

    print("\nBenchmark completed!")
    print(f"Results saved to: {args.out_dir}")
    print("\nSummary of results:")
    for agent_name, result in results.items():
        aggregate = result.get("aggregate", {})
        iqm = aggregate.get("iqm", 0.0)
        num_seeds = aggregate.get("num_seeds", 0)
        print(f"{agent_name}: IQM = {iqm:.3f} (over {num_seeds} seeds)")


if __name__ == "__main__":
    main()
