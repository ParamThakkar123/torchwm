# Benchmarking World Models

This page documents the lightweight benchmarking harness included in the repository.

Quick Overview
--------------
- Code lives under `world_models/benchmarks/`.
- Entrypoints:
  - CLI: `python -m world_models.benchmarks.cli` (see examples below)
  - Python API: `world_models.benchmarks.runner.BenchmarkRunner`

Supported adapters (out of the box)
----------------------------------
- `diamond` - DIAMOND diffusion world-model agent (`world_models.training.train_diamond.DiamondAgent`)
- `iris` - IRIS transformer-based agent (`world_models.training.train_iris.IRISTrainer`)
- `dreamerv1` / `dreamerv2` - Dreamer family (`world_models.models.dreamer.DreamerAgent`)

CLI Examples
------------

- Run IRIS on Pong for 3 episodes using seed 0 (writes results to `results/bench` by default):

```
python -m world_models.benchmarks.cli --agent iris --game ALE/Pong-v5 --seeds 1 --episodes 3
```

- Run DIAMOND on Breakout with two explicit seeds and 5 episodes per seed:

```
python -m world_models.benchmarks.cli --agent diamond --game Breakout-v5 --seeds 0,1 --episodes 5
```

- Run DreamerV2 on a Gym env (example):

```
python -m world_models.benchmarks.cli --agent dreamerv2 --game Pong-v5 --seeds 1 --episodes 10
```

- Run all agents on Pong for 3 episodes using seed 0:

```
python -m world_models.benchmarks.cli --all-agents --game ALE/Pong-v5 --seeds 1 --episodes 3
```

Python API Example
------------------

Use the `BenchmarkRunner` when you need programmatic control:

```py
from world_models.benchmarks.runner import BenchmarkRunner
from world_models.benchmarks import adapters

runner = BenchmarkRunner(adapter_cls=adapters.IRISAdapter, out_dir="results/bench")
res = runner.run(env_spec={"game": "ALE/Pong-v5"}, seeds=[0,1], num_episodes=5)
print(res)
```

Running the Atari 100k Benchmark
---------------------------------

To run the full Atari 100k benchmark on all 26 games using IRIS:

```
python benchmarks/atari_100k.py
```

This will train IRIS on each game for 100k environment steps with 5 random seeds per game, compute human-normalized scores, and compare to baselines.

Outputs
-------
- The runner saves results into the `out_dir` (default `results/bench`):
  - `benchmark_results.json` (raw structured results)
  - `benchmark_results.csv` (seed rows)
  - `benchmark_results.md` (human readable markdown table)
  - `benchmark_results.tex` (LaTeX table ready for papers)

Computing IQM and bootstrap CIs
-------------------------------

The runner stores per-seed means in the JSON under `aggregate.per_seed_means`. Use
the provided metrics helpers to compute IQM and bootstrap confidence intervals:

```py
from world_models.benchmarks import metrics, reporting
import json

res = json.load(open('results/bench/benchmark_results.json'))
per_seed = res['aggregate']['per_seed_means']
iqm = metrics.iqm_of_array(per_seed)
lower, upper = metrics.bootstrap_iqm_ci(per_seed, num_samples=2000, alpha=0.05)
print(f"IQM={iqm:.3f} (95% CI {lower:.3f} - {upper:.3f})")

# export LaTeX table from results
reporting.export_latex(res, 'results/bench/benchmark_results.tex')
```

Extending the harness
---------------------
- Create an adapter in `world_models/benchmarks/adapters.py` that implements:
  - `load_checkpoint(path: str)` and
  - `evaluate(num_episodes: int, render: bool = False)` returning `{"episode_returns": List[float]}`.
- Register your adapter in `world_models/benchmarks/cli.py` to expose it via the CLI.

Tests and CI
-----------
- Place smoke tests under `world_models/benchmarks/tests/` so CI can run them quickly.
- The repo contains a `mocking_classes.py` helper for building fake agents/environments for fast unit tests.

Where to start
--------------
- Run the examples in `examples/benchmark_iris.py` or use the CLI directly.
- If you need help wiring specific agent configs (device, preset, checkpoint paths), use the CLI `--device` and `--preset` flags or call the runner programmatically and pass `extra_kwargs`.
