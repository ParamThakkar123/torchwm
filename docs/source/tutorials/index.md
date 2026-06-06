# Tutorial notebooks

These notebooks provide runnable Python workflows for training each supported TorchWM world-model family on a representative environment or dataset, plus a benchmark notebook for Atari evaluation.

```{toctree}
:maxdepth: 1

notebooks/dreamer_dmc_walker
notebooks/planet_gym_cartpole
notebooks/jepa_cifar10
notebooks/iris_atari_pong
notebooks/genie_tinyworlds_sonic
notebooks/diamond_atari_breakout
notebooks/dit_cifar10
notebooks/atari_benchmark
```

## Recommended order

1. Dreamer on DeepMind Control Walker for online latent-dynamics RL.
2. PlaNet/RSSM on Gym CartPole for planning with a learned latent model.
3. JEPA on CIFAR-10 for self-supervised visual representation learning.
4. IRIS on Atari Pong for transformer world-model control.
5. Genie on TinyWorlds SONIC for controllable video dynamics.
6. DIAMOND on Atari Breakout for diffusion-based world-model RL.
7. DiT/DDPM on CIFAR-10 for diffusion-transformer generation.
8. Atari benchmark notebook to evaluate trained checkpoints.

Most training cells are commented out by default so documentation builds do not launch expensive downloads or training jobs. Uncomment the final training cells after verifying optional dependencies and dataset paths.
