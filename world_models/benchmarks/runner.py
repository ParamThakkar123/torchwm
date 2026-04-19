from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from world_models.benchmarks import adapters, metrics, reporting


class BenchmarkRunner:
    """Run evaluations for adapters across seeds and export results.

    Usage:
        runner = BenchmarkRunner(adapter_cls=adapters.DiamondAdapter)
        results = runner.run(games=["Breakout-v5"], seeds=[0,1], episodes=5)
    """

    def __init__(
        self, adapter_cls: Callable[..., adapters.BaseAdapter], out_dir: str = "results"
    ):
        self.adapter_cls = adapter_cls
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def run(
        self,
        env_spec: Any | None = None,
        seeds: List[int] | None = None,
        num_episodes: int = 5,
        checkpoint: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run benchmark.

        Returns a results dict with per-seed episode returns and computed metrics.
        """
        if checkpoint is None:
            raise ValueError(
                "Checkpoint path is required for benchmarking. Only trained models should be benchmarked."
            )

        seeds = seeds or [0]
        extra_kwargs = extra_kwargs or {}

        all_results: Dict[str, Any] = {
            "seeds": {},
            "aggregate": {},
        }

        per_seed_returns: List[float] = []

        for seed in seeds:
            adapter = self.adapter_cls(env_spec=env_spec, seed=seed, **extra_kwargs)
            if checkpoint:
                try:
                    adapter.load_checkpoint(checkpoint)
                except Exception:
                    # best effort: continue without checkpoint
                    pass

            std_res = adapter.evaluate(num_episodes=num_episodes, render=False)

            # standardize: expect dict with 'episode_returns' or compatible keys
            if isinstance(std_res, dict) and "episode_returns" in std_res:
                ep_returns = list(std_res["episode_returns"])
            elif isinstance(std_res, (list, tuple, np.ndarray)):
                ep_returns = list(std_res)
            elif isinstance(std_res, dict) and "eval_mean_return" in std_res:
                # IRIS trainer returns summary dict by default
                ep_returns = [float(std_res["eval_mean_return"])]
            else:
                # fallback: try to extract numeric values
                ep_returns = []
                for v in std_res.values() if isinstance(std_res, dict) else []:
                    if isinstance(v, (int, float)):
                        ep_returns.append(float(v))

            per_seed_returns.append(float(np.mean(ep_returns) if ep_returns else 0.0))

            all_results["seeds"][str(seed)] = {
                "episode_returns": ep_returns,
                "mean": float(np.mean(ep_returns)) if ep_returns else 0.0,
                "std": float(np.std(ep_returns)) if ep_returns else 0.0,
            }

        # Compute aggregate metrics across seeds
        aggregate: Dict[str, Any] = metrics.compute_aggregate_metrics(per_seed_returns)
        # include raw per-seed means so reporters can compute bootstrap CIs
        aggregate["per_seed_means"] = list(per_seed_returns)
        all_results["aggregate"] = aggregate

        # Save results
        out_json = os.path.join(self.out_dir, "benchmark_results.json")
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)

        # Export pretty table
        reporting.export_csv(
            all_results, os.path.join(self.out_dir, "benchmark_results.csv")
        )
        reporting.export_markdown(
            all_results, os.path.join(self.out_dir, "benchmark_results.md")
        )
        reporting.export_latex(
            all_results,
            os.path.join(self.out_dir, "benchmark_results.tex"),
            caption="Benchmark results",
        )

        return all_results


class MultiAgentBenchmarkRunner:
    """Run evaluations for multiple adapters on the same environment.

    Usage:
        runner = MultiAgentBenchmarkRunner(adapters=[adapters.DiamondAdapter, adapters.IRISAdapter])
        results = runner.run_all(game="Breakout-v5", seeds=[0,1], episodes=5)
    """

    def __init__(
        self,
        adapters: List[Callable[..., adapters.BaseAdapter]],
        out_dir: str = "results",
    ):
        self.adapters = adapters
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def run_all(
        self,
        env_spec: Any | None = None,
        seeds: List[int] | None = None,
        num_episodes: int = 5,
        checkpoints: Optional[Dict[str, str]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run benchmarks for all adapters on the same environment.

        Returns a results dict with results for each adapter.
        """
        seeds = seeds or [0]
        extra_kwargs = extra_kwargs or {}
        if checkpoints is None:
            raise ValueError(
                "Checkpoints dict is required for benchmarking. Only trained models should be benchmarked."
            )
        if not checkpoints:
            raise ValueError(
                "Checkpoints dict cannot be empty. Provide checkpoint paths for all agents."
            )

        all_results: Dict[str, Any] = {}

        for adapter_cls in self.adapters:
            adapter_name = adapter_cls.__name__.replace("Adapter", "").lower()
            if adapter_name not in checkpoints:
                raise ValueError(
                    f"Checkpoint path required for {adapter_name}. Only trained models should be benchmarked."
                )
            checkpoint = checkpoints[adapter_name]

        all_results: Dict[str, Any] = {}

        for adapter_cls in self.adapters:
            adapter_name = adapter_cls.__name__.replace("Adapter", "").lower()
            print(f"Running benchmark for {adapter_name}...")

            runner = BenchmarkRunner(
                adapter_cls=adapter_cls,
                out_dir=os.path.join(self.out_dir, adapter_name),
            )
            checkpoint = checkpoints.get(adapter_name)

            result = runner.run(
                env_spec=env_spec,
                seeds=seeds,
                num_episodes=num_episodes,
                checkpoint=checkpoint,
                extra_kwargs=extra_kwargs,
            )

            all_results[adapter_name] = result

        # Save combined results
        combined_json = os.path.join(self.out_dir, "combined_benchmark_results.json")
        with open(combined_json, "w") as f:
            json.dump(all_results, f, indent=2)

        # Export combined CSV
        self._export_combined_csv(
            all_results, os.path.join(self.out_dir, "combined_benchmark_results.csv")
        )

        return all_results

    def _export_combined_csv(self, results: Dict[str, Any], filepath: str):
        """Export combined results to CSV."""
        import csv

        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Agent", "Seed", "Mean Return", "Std Return"])

            for agent, data in results.items():
                for seed, seed_data in data.get("seeds", {}).items():
                    writer.writerow(
                        [
                            agent,
                            seed,
                            seed_data.get("mean", 0.0),
                            seed_data.get("std", 0.0),
                        ]
                    )


if __name__ == "__main__":
    # example quick-run with mocks (user should use CLI)
    from world_models.benchmarks.adapters import IRISAdapter

    runner = BenchmarkRunner(adapter_cls=IRISAdapter, out_dir="results/bench")
    res = runner.run(env_spec={"game": "ALE/Pong-v5"}, seeds=[0], num_episodes=2)
    print(res)
