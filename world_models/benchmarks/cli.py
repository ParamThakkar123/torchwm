from __future__ import annotations

import argparse
from typing import List
import torch

from world_models.benchmarks.runner import BenchmarkRunner, MultiAgentBenchmarkRunner
from world_models.benchmarks import adapters


AGENTS = {
    "diamond": adapters.DiamondAdapter,
    "iris": adapters.IRISAdapter,
    "dreamerv1": adapters.DreamerV1Adapter,
    "dreamerv2": adapters.DreamerV2Adapter,
}


def parse_seeds(s: str) -> List[int]:
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    if s.isdigit():
        return list(range(int(s)))
    return [int(s)]


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark for an agent on an environment"
    )
    parser.add_argument(
        "--agent", type=str, choices=list(AGENTS.keys()), required=False
    )
    parser.add_argument(
        "--all-agents", action="store_true", help="Run benchmark for all agents"
    )
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument(
        "--seeds",
        type=str,
        default="1",
        help="Either N (creates seeds 0..N-1) or comma list",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/bench")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--train-epochs", type=int, default=None)

    args = parser.parse_args()

    if not args.all_agents and not args.agent:
        parser.error("Either --agent or --all-agents must be specified")

    if args.all_agents and args.agent:
        parser.error("--agent and --all-agents cannot be used together")

    if args.agent and args.checkpoint is None:
        parser.error(
            "--checkpoint is required when using --agent. Only trained models should be benchmarked."
        )

    seeds = parse_seeds(args.seeds)

    if args.all_agents:
        # Run all agents on the same game
        all_adapters = list(AGENTS.values())
        runner = MultiAgentBenchmarkRunner(adapters=all_adapters, out_dir=args.out_dir)

        res = runner.run_all(
            env_spec={"game": args.game},
            seeds=seeds,
            num_episodes=args.episodes,
            checkpoints=None,  # Could extend to support per-agent checkpoints
            extra_kwargs={"device": args.device, "preset": args.preset},
            train_epochs=args.train_epochs,
        )

        print("Multi-agent benchmark finished. Results written to:", args.out_dir)
    else:
        # Run single agent
        adapter_cls = AGENTS[args.agent]

        runner = BenchmarkRunner(adapter_cls=adapter_cls, out_dir=args.out_dir)

        env_spec = {"game": args.game}

        res = runner.run(
            env_spec=env_spec,
            seeds=seeds,
            num_episodes=args.episodes,
            checkpoint=args.checkpoint,
            extra_kwargs={"device": args.device, "preset": args.preset},
        )

        print("Benchmark finished. Results written to:", args.out_dir)


if __name__ == "__main__":
    main()
