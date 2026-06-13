TorchWM CLI
===========

The project exposes a small command-line interface for common developer tasks:

- Run the CLI with: `python -m tools.cli <command>`; after installing the
  package an installed entrypoint is available as `torchwm <command>`. Tests
  and plugin integrations may invoke the top-level Click app (`tools.cli.app`)
  or the console-script callable (`tools.cli.run`).
- The CLI uses Click directly and lazy imports to keep startup fast; some
  commands require optional dependencies (listed below).

Commands
--------

- `version` - Print the installed `torchwm` package version (or "unknown" if
  the package cannot be imported).

- `envs list` - List built-in environment backends and example environment ids.
  This reads the environment catalog from `world_models.catalog` if available.

- `datasets list [PATH]` - List dataset entries under `PATH`. If `PATH` is not
  provided the command uses `TORCHWM_HOME` or defaults to `~/.torchwm`.

- `datasets convert <src> [--dest-format video] [--out-dir DIR]` - Convert a
  simple dataset file into another format. The initial implementation supports
  converting HDF5 (`.h5`) or NumPy (`.npz`/`.npy`) datasets into MP4 video
  files (one file per episode) when `--dest-format video` is used. Output files
  are written to the specified `--out-dir` or `./converted_datasets` by
  default.

- `collect --env <ENV_ID> [--steps N] [--out PATH] [--random-policy]` - Run a
  (random) policy for a number of environment steps and save interactions to
  a compressed `.npz` with keys `observations`, `actions`, `rewards`,
  `dones`.

- `train <model> [extra args...] [--inproc]` - Launch an existing training
  entrypoint. The CLI maps simple model names to modules in
  `world_models.training` (e.g. `diamond`, `iris`, `planet`, `jepa`, `rssm`,
  `genie`). By default `train` spawns a subprocess running
  `python -m world_models.training.<name>` and forwards any extra args. Use
  `--inproc` to attempt running the training entrypoint in-process (calls the
  module's `main()` if available). DIAMOND, IRIS, and JEPA accept
  `--config PATH`, `--print-config`, and OmegaConf/Hydra-style dot-list
  overrides such as `total_epochs=100` or `optimization.lr=3e-4`.

- `models list` - Print the known training entrypoints and (when available)
  exported model names from `world_models.models`.

Environment / optional dependencies
----------------------------------

- TORCHWM_HOME - Directory used by `datasets list` when no path is provided.

- The following commands require optional packages which may not be installed
  in all environments:
  - `collect`: requires `gym`/`gymnasium` and `numpy`.
  - `datasets convert`: requires `h5py`, `numpy` and video helpers used by the
    repository.

Notes and examples
------------------

- Example: show version

```bash
torchwm version
```

- Example: list environments

```bash
torchwm envs list
```

- Example: list datasets in default location

```bash
torchwm datasets list
```

- Example: convert a local HDF5 dataset to MP4 files

```bash
torchwm datasets convert data/my_dataset.h5 --out-dir /tmp/videos
```

- Example: collect 1000 steps from Pong and save as `pong.npz`

```bash
torchwm collect --env ALE/Pong-v5 --steps 1000 --out pong.npz
```

- Example: run IRIS training with a library YAML config and a dot-list override

```bash
torchwm train iris --config world_models/configs/experiments/iris.yaml total_epochs=100
```

- Example: inspect a composed JEPA config without starting training

```bash
torchwm train jepa --config world_models/configs/experiments/jepa.yaml optimization.epochs=50 --print-config
```

- Example: launch a DIAMOND preset from the unified training CLI

```bash
torchwm train diamond --config world_models/configs/experiments/diamond.yaml preset=small seed=3
```

Maintaining this page
---------------------

If you add or rename CLI commands in `tools.cli`, update this page with the
new usage, examples, and any additional optional dependencies.
