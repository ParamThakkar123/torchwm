# Benchmarking World Models

TorchWM includes a lightweight benchmark harness for running standardized
evaluations of trained world-model agents and exporting results that can be used
in experiment logs, reports, and papers.

## Quick Overview

- Code lives under `world_models/benchmarks/`.
- Preferred CLI entrypoint: `torchwm benchmark`.
- Module entrypoint: `python -m world_models.benchmarks.cli`.
- Python API: `world_models.benchmarks.runner.BenchmarkRunner`.

## Supported adapters

The benchmark CLI currently registers these adapters out of the box:

- `diamond` - DIAMOND diffusion world-model agent (`world_models.training.train_diamond.DiamondAgent`)
- `iris` - IRIS transformer-based agent (`world_models.training.train_iris.IRISTrainer`)
- `dreamerv1` / `dreamerv2` - Dreamer family (`world_models.models.dreamer.DreamerAgent`)

Benchmarks are intended for trained models. For single-agent runs, pass a
checkpoint with `--checkpoint`. For multi-agent runs, pass one or more
`--checkpoint-map AGENT=PATH` values, or use `--train-epochs` when you
intentionally want the CLI to train before evaluating.

## TorchWM CLI examples

Run IRIS on Pong for 3 episodes using seed 0 and write the standard result files
to `results/bench`:

```bash
torchwm benchmark \
  --agent iris \
  --game ALE/Pong-v5 \
  --checkpoint checkpoints/iris/pong.pt \
  --seeds 1 \
  --episodes 3
```

Run DIAMOND on Breakout with two explicit seeds and 5 episodes per seed:

```bash
torchwm benchmark \
  --agent diamond \
  --game Breakout-v5 \
  --checkpoint checkpoints/diamond/breakout.pt \
  --seeds 0,1 \
  --episodes 5 \
  --out-dir results/diamond_breakout
```

Run DreamerV2 on a Gym environment:

```bash
torchwm benchmark \
  --agent dreamerv2 \
  --game Pong-v5 \
  --checkpoint checkpoints/dreamerv2/pong.pt \
  --seeds 1 \
  --episodes 10 \
  --device cpu
```

Run all registered adapters on the same game with per-agent checkpoints:

```bash
torchwm benchmark \
  --all-agents \
  --game ALE/Pong-v5 \
  --checkpoint-map iris=checkpoints/iris/pong.pt \
  --checkpoint-map diamond=checkpoints/diamond/pong.pt \
  --checkpoint-map dreamerv1=checkpoints/dreamerv1/pong.pt \
  --checkpoint-map dreamerv2=checkpoints/dreamerv2/pong.pt \
  --seeds 0,1 \
  --episodes 5 \
  --out-dir results/pong_comparison
```

## CLI options

Common `torchwm benchmark` options:

- `--agent AGENT` / `-a AGENT`: run one adapter (`iris`, `diamond`, `dreamerv1`, or `dreamerv2`).
- `--all-agents`: run every registered adapter on the same environment.
- `--game GAME` / `-g GAME`: Gym/ALE environment id, such as `ALE/Pong-v5`.
- `--checkpoint PATH` / `-c PATH`: checkpoint path for single-agent benchmarks.
- `--checkpoint-map AGENT=PATH`: repeatable per-agent checkpoint mapping for `--all-agents`.
- `--seeds SEEDS`: either `N` for seeds `0..N-1`, or a comma-separated list such as `0,1,2`.
- `--episodes N` / `-n N`: number of evaluation episodes per seed.
- `--out-dir DIR`: output directory for report artifacts. The legacy alias `--out_dir` is also accepted.
- `--device DEVICE`: device forwarded to adapters. Defaults to CUDA when available, otherwise CPU.
- `--preset PRESET`: optional adapter/model preset.
- `--train-epochs N`: for `--all-agents`, train first when checkpoint maps are not supplied.

You can also run `torchwm benchmark --help` to see the installed CLI help.

## Module CLI compatibility

The lower-level module entrypoint remains available for scripts that already use
it:

```bash
python -m world_models.benchmarks.cli \
  --agent iris \
  --game ALE/Pong-v5 \
  --checkpoint checkpoints/iris/pong.pt \
  --seeds 1 \
  --episodes 3
```

For new usage, prefer `torchwm benchmark` so benchmark runs are discoverable
through the main TorchWM CLI alongside `torchwm train`, `torchwm envs`, and
`torchwm datasets`.

## Python API example

Use `BenchmarkRunner` when you need programmatic control:

```py
from world_models.benchmarks.runner import BenchmarkRunner
from world_models.benchmarks import adapters

runner = BenchmarkRunner(adapter_cls=adapters.IRISAdapter, out_dir="results/bench")
res = runner.run(
    env_spec={"game": "ALE/Pong-v5"},
    seeds=[0, 1],
    num_episodes=5,
    checkpoint="checkpoints/iris/pong.pt",
)
print(res)
```

## Running the Atari 100k benchmark

To run the full Atari 100k benchmark on all configured games with the
centralized benchmark module:

```bash
python -m world_models.benchmarks.atari_100k --benchmark
```

This runs the Atari 100k evaluator from `world_models/benchmarks`, computes
human-normalized scores, and reports aggregate metrics across games and seeds.

## Outputs

The runner saves these files into the selected `out_dir` (default
`results/bench`):

- `benchmark_results.json` - raw structured results.
- `benchmark_results.csv` - per-seed rows.
- `benchmark_results.md` - human-readable markdown table.
- `benchmark_results.tex` - LaTeX table ready for papers.

Multi-agent runs also write combined reports such as
`combined_benchmark_results.json` and `combined_benchmark_results.csv` in the
root output directory, with per-agent details under subdirectories.

## Computing IQM and bootstrap CIs

The runner stores per-seed means in the JSON under
`aggregate.per_seed_means`. Use the provided metrics helpers to compute IQM and
bootstrap confidence intervals:

```py
from world_models.benchmarks import metrics, reporting
import json

res = json.load(open("results/bench/benchmark_results.json"))
per_seed = res["aggregate"]["per_seed_means"]
iqm = metrics.iqm_of_array(per_seed)
lower, upper = metrics.bootstrap_iqm_ci(per_seed, num_samples=2000, alpha=0.05)
print(f"IQM={iqm:.3f} (95% CI {lower:.3f} - {upper:.3f})")

reporting.export_latex(res, "results/bench/benchmark_results.tex")
```

## Extending the harness

- Create an adapter in `world_models/benchmarks/adapters.py` that implements:
  - `load_checkpoint(path: str)`
  - `evaluate(num_episodes: int, render: bool = False)` returning `{"episode_returns": list[float]}`
- Register your adapter in `world_models/benchmarks/cli.py` to expose it through both `python -m world_models.benchmarks.cli` and `torchwm benchmark`.

## Tests and CI

- Place smoke tests under `world_models/benchmarks/tests/` so CI can run them quickly.
- The repo contains a `mocking_classes.py` helper for building fake agents and environments for fast unit tests.

## Where to start

- Run the examples in `examples/benchmark_iris.py` or use `torchwm benchmark` directly.
- If you need help wiring specific agent configs, use `--device`, `--preset`, and checkpoint options, or call the runner programmatically and pass `extra_kwargs`.
