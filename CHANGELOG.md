# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Fixes from the September 2026 code audit.

### Fixed
- IRIS evaluation collected every raw frame and a per-step latent even with
  `render=False`, the default during training, and then discarded them. At the
  default `eval_episodes=100` and the 27000-step episode cap that is tens of GB
  held until the evaluation returns, so the OS could kill the process outright
  (in a notebook, the kernel just restarts). Both are now render-only
- IRIS evaluation reset the same environment `collect_experience` keeps a
  partial episode on, so the next collection step paired the stale pre-eval
  observation with the post-eval environment and wrote a transition that never
  happened into the replay buffer. Evaluation now runs on its own environment;
  when the caller supplied the environment, the in-flight collection episode is
  dropped instead
- Genie training on TinyWorlds could kill the process with no traceback (a
  notebook kernel just restarts) whenever `num_workers > 0`, the default:
  `TinyWorldsDataset` kept one HDF5 handle open from `__init__`, and forked
  DataLoader workers all read through that inherited handle, which HDF5 does
  not support. Each process now opens its own handle on first read. The same
  handle made spawned workers (Windows, macOS) fail with
  `TypeError: h5py objects cannot be pickled`
- System-metrics logging crashed training on GPU machines: `collect_system_stats`
  guarded `torch.cuda.utilization` with `hasattr`, which is always true, while
  calling it raises `ModuleNotFoundError` without NVML (`nvidia-ml-py`). The
  counter is now skipped with a single warning, and `nvidia-ml-py` is part of the
  `ml` extra
- Dreamer acted on the RSSM prior instead of the posterior, so the current
  observation was encoded and then ignored during collection and evaluation
  (also in `play dreamer` and `scripts/benchmark_infer.py`)
- Dreamer checkpoints omitted the critic (`value_model`); older checkpoints
  still load, keeping a fresh critic and skipping its optimizer state
- `Dreamer.evaluate(render=True)` raised `KeyError: 0`
- `use_disc_model=True` raised `TypeError` in the actor loss; the discount head
  output is now also scaled by `discount`, as in the reference implementation
- Time-limit truncations were stored as terminal transitions; the replay buffer
  now keeps episode boundaries and true terminations separately
- `DreamerConfig.env_instance` made startup crash while writing `config.yaml`
- `Dreamer(restore=...)` ignored its argument
- `DreamerAgent.train()` now always writes a final checkpoint
- PlaNet sampled latents with uniform instead of Gaussian noise; `rollout_prior`
  returned tuples; `get_init_state` conditioned the posterior on the stale
  deterministic state
- Genie's default 5120-wide dynamics model used 36 heads, which do not divide
  the width (now 40); attention layers validate `dim % num_heads`
- `create_model("genie"/"genie-small")` silently dropped tokenizer and
  latent-action widths/depths from the config; `tokenizer_num_heads` and
  `action_num_heads` are now honoured (default 16, which is what was always
  built)
- DIAMOND collection kept every frame of the current episode in memory
- `torchwm train --inproc` re-ran training without arguments on `TypeError`
  and in a subprocess on any other error
- `torchwm eval` / `torchwm play` imported modules that are not in the wheel;
  they now live in `torchwm.inference`, and the metrics package moved from the
  top-level `evals` to `torchwm.evals`
- `register_env_backend` backends were never used by `make_env`; a `dmc`
  backend was added
- Every `torch.load` in scripts and demos now uses `weights_only=True`
- IRIS play in `scripts/benchmark_infer.py` reset the policy LSTM every step
- MP4 video logging wrapped uint8 frames around; GIFs are written with Pillow
  instead of the undeclared moviepy
- `MUJOCO_GL=egl` is only defaulted on Linux (macOS has no EGL)
- A broken W&B install no longer breaks importing JEPA training

### Added
- `python -m torchwm.install_dmc` (also `make install-dmc`) installs the
  DeepMind Control backend on CPython 3.13, where dm-control's `labmaze`
  dependency has no wheel and builds with Bazel. It uses the pure-Python
  `labmaze-new` and installs dm-control with `--no-deps`. `--check` verifies an
  environment, `--dry-run` prints the commands. It keeps the installed mujoco and
  matches dm-control to it, because mujoco 3.13 removed an enum mujoco-mjx still
  uses and upgrading would break the `brax` extra; `--upgrade-mujoco` opts out.
  `[tool.uv] constraint-dependencies` caps `mujoco<3.13` for the same reason
- CUDA, then Apple MPS, then CPU device selection
  (`torchwm.utils.device`)
- `worldmodels` extra (albumentations, cma); `hydra-core`/`omegaconf` in `ml`;
  `h5py` in `viz`; `ruff` in `dev`

### Removed
- The `procgen` extra, which could never install on Python >= 3.11
- `dm-control` from the `dmc` extra on CPython 3.13 only, so
  `pip install torchwm[dmc]` no longer fails there. New `dmc-uv` extra keeps it
  unconditional for uv, which drops the `labmaze` pin through
  `[tool.uv] override-dependencies`
- `tools` is no longer installed as a top-level package
- `nginx.conf`, which proxied a frontend that no longer exists

## [1.0.0] — 2026-08-19

First stable release. The public API surface documented in
`docs/source/public_api.md` is now covered by semantic versioning: breaking
changes to it require a major version bump and a deprecation cycle.

Because 0.5.0 below was never tagged or published, upgrading from 0.4.2 — the
last release on PyPI — also brings in every 0.5.0 change.

### Known limitations
- `dreamer-v3` is a registry name for `DreamerAgent`, not a paper-complete
  DreamerV3 implementation
- Shared step-budget `train()` covers the Dreamer family; other models use
  their own trainers or `torchwm train`
- On CPython 3.13 the `dmc` extra installs everything except `dm_control`
  itself; run `python -m torchwm.install_dmc` to add it (see the Unreleased
  section)

### Added
- Throughput instrumentation: `torchwm.ThroughputMeter`, `torchwm.measure_steps`
  and `torchwm.tensor_nbytes` report steps/sec, ms/step and bytes shipped to the
  device per step, synchronising on CUDA so the timer measures execution rather
  than queueing
- Performance helpers: `torchwm.enable_performance_defaults` (cuDNN autotuning
  and TF32, no-ops without CUDA), `torchwm.maybe_compile` (opt-in
  `torch.compile` with an eager fallback) and `torchwm.to_channels_last`
- `GenieConfig.use_amp` / `GenieSmallConfig.use_amp` — autocast for Genie
  training, preferring bfloat16 where supported so no gradient scaler is needed
- Genie `VideoDataset` loads `.npy` / `.npz` / `.pt` clips, or video files when
  OpenCV (`torchwm[viz]`) is installed
- `DreamerConfig.perf_defaults` and `DreamerConfig.tf32`
- `RSSMPolicy(..., compile_rollout=True)` compiles the CEM candidate-rollout
  step, which runs `num_iterations * planning_horizon` tiny kernels per env step
- I-JEPA linear evaluation (`torchwm.training.eval_jepa`, exported as
  `torchwm.jepa_linear_probe` / `torchwm.load_jepa_encoder`), implementing the
  paper's Appendix A.2 protocol: frozen target-encoder, average-pooled patch
  tokens, LARS-trained linear head, and the published sweep over learning rate,
  weight decay, batch-norm head, and last-layer vs last-four-layer features
- `VisionTransformer.get_intermediate_layers()`, needed for the last-four-layer
  probe representation
- `torchwm/configs/experiments/jepa_small_gpu.yaml` — single-GPU preset
  that reduces only batch size, backbone, and epochs, leaving every method
  parameter at the paper's value
- `tests/models/test_jepa_paper_alignment.py`, pinning the I-JEPA masking,
  architecture, loss, and schedule details against the paper
- `torchwm eval --model jepa` runs the linear probe from the CLI, alongside the
  existing `--model diamond` FID/FVD/LPIPS path. The two evaluations share only
  `--checkpoint`, `--batch-size`, `--device` and `--output`; passing an option
  that belongs to the other model is an error rather than a silent no-op

### Changed
- **Breaking: one package.** The implementation moved from `world_models/` into
  `torchwm/`, and the alias layer that made `torchwm.<name>` resolve to
  `world_models.<name>` is gone. `torchwm` is now the only import path —
  `import world_models` raises `ModuleNotFoundError`. Replace
  `from world_models.x import y` with `from torchwm.x import y`; the public
  `torchwm` surface is unchanged
- **I-JEPA defaults now reproduce the paper.** The shipped configuration
  previously combined the two worst settings in the paper's own ablations:
  `enc_mask_scale` was `(0.15, 0.2)` where the context block calls for
  `(0.85, 1.0)` (Table 9), and `num_pred_masks` was `1` where the paper uses `4`
  (Table 10: 9.0 vs 54.2 low-shot top-1). View augmentations were also enabled
  by default, contradicting the paper's central claim; `use_gaussian_blur`,
  `use_horizontal_flip` and `use_color_distortion` now default to `False`.
  `min_keep` is `10`, `crop_scale` is `(0.3, 1.0)`, and the optimizer follows
  Appendix A: batch 2048, warmup 1e-4 -> 1e-3 over 15 epochs, cosine to 1e-6
- I-JEPA learning rates are quoted at the paper's batch size of 2048 and scaled
  linearly to the effective batch size; set `lr_reference_batch_size = None` to
  opt out
- I-JEPA trains with the paper's L2 loss (`loss_type="l2"`) instead of the
  reference implementation's Smooth-L1, which remains available as
  `loss_type="smooth_l1"` alongside the literal per-block sum `"l2_sum"`
- `JEPAConfig.pred_depth` defaults to `None`, which selects the paper's
  predictor depth for the configured backbone (6 for ViT-B, 12 for ViT-L/H, 16
  for ViT-G) instead of silently building a 6-layer predictor for every model
- The multi-block mask sampler draws block scale and aspect ratio from
  independent uniforms, as Sec. 3 specifies, rather than sharing one draw
- No `GradScaler` is created for bfloat16 training, which does not need loss
  scaling; `torch.autocast` replaces the deprecated `torch.cuda.amp` entry points
- Dreamer keeps replay observations in `uint8` across the host-to-device copy.
  The buffer already stores `uint8` and `preprocess_obs` already casts on the
  device, so the old `torch.tensor(obs, dtype=torch.float32)` widened the batch
  on the host and moved four times the bytes for a cast that happened anyway —
  123MB per step at stock config, now 30.7MB. Transfers use `torch.from_numpy`
  (no extra host copy) and pinned, non-blocking staging on CUDA
- The ViT/I-JEPA backbone uses `scaled_dot_product_attention` instead of an
  explicit `softmax(q @ k^T * scale) @ v`, so the (B, heads, N, N) score matrix
  is never materialised. Every other attention block in the package already did.
  A custom `qk_scale` is passed through unchanged; outputs match the explicit
  form to 1.7e-16 in float64
- Target-network EMA updates (I-JEPA, DiT) use `torch._foreach_*`: one fused
  multi-tensor op per step rather than two kernels per parameter tensor, and
  `alpha=` applies the `(1 - m)` scale inside the add instead of allocating a
  scaled copy of every source tensor. Results differ by at most one ulp, in the
  fused form's favour — it rounds once where the explicit form rounded twice
- Dataloaders pin host memory when CUDA is present and keep workers alive across
  epochs. `tinyworlds` hardcoded `pin_memory=False`, and the ImageNet loaders
  respawned every worker at each epoch boundary
- FID/LPIPS/FVD share one frozen backbone per device instead of rebuilding (and
  re-downloading) Inception or VGG for every metric instance
- The PlaNet CEM planner runs under `torch.inference_mode` rather than
  `no_grad`, and no longer clones state before broadcasting it over candidates
- The Dreamer replay buffer's sequence sampler tests the episode-boundary
  rejection arithmetically instead of materialising the index window for every
  rejected draw. The RNG is consumed identically, so a given seed still yields
  exactly the same sequences
- uv no longer resolves the whole project through a CUDA-specific PyTorch index.
  `[tool.uv]` set `https://download.pytorch.org/whl/cu121` as the *default*
  index with `index-strategy = "unsafe-best-match"`, which routed every package
  through it and pinned contributors to one CUDA build regardless of hardware —
  contradicting the README. Resolution now comes from PyPI; add the index for
  your platform explicitly if you need a specific wheel set

### Removed
- **Breaking:** the `torchwm.inference` operator
  package (`get_operator`, `OperatorABC`, `TensorSpec`, `DreamerOperator`,
  `JEPAOperator`, `IrisOperator`, `PlaNetOperator`). The operators only resized
  and normalized tensors, and `JEPAOperator` masked uniformly at random, which
  is not I-JEPA's masking at all. Preprocess inputs directly, or use
  `torchwm.transforms.image.make_transforms` and
  `torchwm.masks.MultiblockMaskCollator`
- **Breaking:** the unused `operator_state_dim` / `operator_action_dim` fields
  on `DiamondConfig`
- **Breaking:** the `minerl` and `minedojo` extras, and the `selenium` extra.
  Neither Minecraft extra could ever install — MineRL 1.x has no Python 3.11+
  release and MineDojo pins `gym==0.21.0`, whose sdist no longer builds — and
  between them they made `uv lock` unresolvable for the whole project.
  `torchwm.envs.minecraft_env` is unchanged; `docs/source/iris.md` documents the
  manual Python 3.10 install

### Fixed
- The multi-block mask sampler raised no error when `min_keep` exceeded the
  patches a block can hold; it now fails with an explanatory message instead of
  looping forever
- The test suite can run to completion in a single process again. Every
  `nn.Module` holds reference cycles, so a model built in a test is reclaimable
  only by CPython's cyclic collector — which triggers on allocation *counts*,
  not bytes, so a handful of very large tensors never trips it. The suite
  retained ~4.4GB (1.6GB from `tests/models/test_genie.py`, 1.4GB from
  `tests/evals/test_evals.py`) and died partway through with either
  `RuntimeError: can't start new thread` or a Windows access violation. A new
  `tests/conftest.py` collects after each test; the suite now finishes, at about
  15% more wall-clock

## [0.5.0] — 2026-07-27 (never tagged or published)

### Added
- Real PEP 561 typing stub: `torchwm/__init__.pyi` now declares every public
  export instead of falling back to `Any`, generated from the export map by
  `python -m tools.gen_type_stub` and verified in CI. `torchwm` ships a
  `py.typed` marker so the re-exports stay typed
- `torchwm play -m dreamer` — interactive REAL/DREAM playback for Dreamer
  checkpoints alongside DIAMOND (`scripts/play_dreamer.py`)
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1)
- CI job running the full test suite with every optional backend installed —
  the configuration a contributor gets from `CONTRIBUTING.md`, previously
  untested
- Tests asserting every `__all__` entry resolves, that the stub matches the
  export map, and that the README algorithms table matches the model registry
- Complete `torchwm` public import surface: every implementation submodule is now
  reachable through the friendly namespace (`from torchwm.models import Dreamer`,
  `import torchwm.envs`, ...), not just top-level factory helpers
- `dmc` optional-dependency extra (`pip install torchwm[dmc]`) that installs
  `dm-control` for the default DeepMind Control backend
- Actionable error from the DMC backend that names the missing `dm_control`
  dependency and points to `torchwm[dmc]` or the gym backend

### Changed
- README "Supported Algorithms" table now lists all 13 registered models
  (previously 5), keyed by the name `create_model()` accepts
- `viz` extra installs only what it uses (opencv, umap-learn, scikit-learn,
  plotly); the unused FastAPI/uvicorn/starlette/python-multipart entries and the
  duplicated docs dependencies are gone
- CI runs the test matrix on `push` to `main` only, not on every push to a PR
  branch, which ran the whole matrix twice per commit
- Documentation, examples, and scripts now import through the `torchwm` public
  namespace
- Quick-start examples (README, docs landing page, getting-started guide) now use
  the base-installable `Pendulum-v1` gym backend so they run on
  `pip install torchwm[gym]` out of the box; DMC usage is documented separately

### Fixed
- `torchwm.DreamerRSSM` and `torchwm.MujocoEnv` raised `AttributeError` on
  access: the export map pointed `DreamerRSSM` at a module that names the class
  `RSSM`, and `MujocoEnv` had no implementation behind it (it now resolves to
  `MuJoCoImageEnv`)
- README advertised a FastAPI visualization feature that does not exist
- `create_model("dreamer", env="walker-walk")` no longer raises a bare
  `ModuleNotFoundError`; the DMC dependency is installable and the error is clear
- Broken landing-page example that called `train(env_name=..., total_steps=...)`
  (the `train` method only accepts `total_steps`)
- Env-adapter tests patched non-existent module attributes
  (`torchwm.models.dreamer.env_wrapper.*`) and the wrong Gymnasium registry
  reference, causing 10 spurious failures
- Removed stray build artifacts (`nul`, empty `testsdata/` directories) and added
  `.gitignore` guards

## [0.4.2] — 2026-06-20

### Added
- Dreamer integration test for Pendulum-v1 wired into CI
- DIAMOND world model documentation
- `from_pretrained` and `from_config` class methods for Dreamer, IRIS, JEPA, Genie agents
- Gymnasium wrapper for world model environment
- ONNX export support for agents
- `py.typed` marker for PEP 561 type declarations
- PSNR evaluation metric

### Changed
- Restructured dreamer docs with separate V1/V2 theory and examples
- Stripped base deps to minimum, moved extras to optional groups
- Centralized version in `_version.py` as single source of truth
- Improved block exports and testing utilities

### Fixed
- Arbitrary code execution risk in pickle.load (hardened checkpoint/replay deserialization)
- Zero-element tensor reshape crash in ConvEncoder
- Empty sequence edge case in Dreamer training
- 25 GitHub Dependabot vulnerabilities (upgraded 8 packages)
- Dockerfile referencing removed `torchwm_ui` folder
- Replaced debug print calls with proper logging
- MyPy type errors across 40+ files
- Missing imports in `train_jepa.py` (mp, F, DistributedDataParallel)
- CI workflow and docs dependency config
- DreamerConfig documentation field sync

## [0.4.1] — 2026-06-01

### Added
- Modular RSSM with swappable LSTM/Transformer/MLP backbones
- Genie model support (video tokenizer, latent action model, dynamics model)
- Brax environment backend
- BSuite environment backend
- DMLab environment backend
- Procgen environment backend
- Robotics environment backend (gymnasium-robotics)
- Unity ML-Agents environment backend
- Sphinx documentation with auto-deploy to GitHub Pages
- Benchmark runners and reporting utilities
- CLI tools (`torchwm`, `torchwm-train`)

### Changed
- Migrated configs to dataclass style for consistency
- Improved lazy import architecture for faster CLI startup

### Fixed
- Cross-platform memory detection (Windows ctypes + Linux /proc/meminfo + psutil fallback)
- Environment wrapper stack consistency across backends

## [0.4.0] — 2026-05-15

### Added
- Initial public release
- Dreamer (V1/V2) agent implementation
- PlaNet agent implementation
- JEPA self-supervised learning agent
- IRIS sample-efficient RL agent
- DiT (Diffusion Transformer) support
- DIAMOND diffusion world model for Atari
- Core environment backends (DMC, Gym, Atari, MuJoCo)
- Replay buffers (Dreamer, IRIS, PlaNet)
- VQ-VAE and ConvVAE vision components
- HuggingFace Hub checkpoint loading
- TensorBoard and WandB logging integration
