# Tutorial notebooks

These notebooks provide runnable Python workflows for training each supported TorchWM world-model family on a representative environment or dataset, plus a benchmark notebook for Atari evaluation.

The notebooks are linked as downloadable `.ipynb` files instead of rendered pages so the documentation site can build in minimal environments that do not install notebook-rendering extensions. Open a notebook locally with Jupyter or VS Code, then uncomment the long-running training cells when you are ready to run them.

## Notebook downloads

- {download}`Dreamer on DeepMind Control Walker <notebooks/dreamer_dmc_walker.ipynb>`
- {download}`PlaNet/RSSM on Gym CartPole <notebooks/planet_gym_cartpole.ipynb>`
- {download}`JEPA on CIFAR-10 <notebooks/jepa_cifar10.ipynb>`
- {download}`IRIS on Atari Pong <notebooks/iris_atari_pong.ipynb>`
- {download}`Genie on TinyWorlds SONIC <notebooks/genie_tinyworlds_sonic.ipynb>`
- {download}`DIAMOND on Atari Breakout <notebooks/diamond_atari_breakout.ipynb>`
- {download}`DiT/DDPM on CIFAR-10 <notebooks/dit_cifar10.ipynb>`
- {download}`Run a trained model on the Atari benchmark <notebooks/atari_benchmark.ipynb>`

## Managed notebook runtimes

If you are running these notebooks on Kaggle, Colab, or another image with preinstalled CUDA/PyTorch packages, install only the extras needed by the notebook instead of forcing every TorchWM optional dependency into the runtime. For example, the Dreamer DMC notebook includes a managed-runtime recipe that uses `pip install --no-deps torchwm` followed by the DMC backend packages, then asks you to restart the kernel before importing TorchWM.

## Recommended order

1. Dreamer on DeepMind Control Walker for online latent-dynamics RL.
2. PlaNet/RSSM on Gym CartPole for planning with a learned latent model.
3. JEPA on CIFAR-10 for self-supervised visual representation learning.
4. IRIS on Atari Pong for transformer world-model control.
5. Genie on TinyWorlds SONIC for controllable video dynamics.
6. DIAMOND on Atari Breakout for diffusion-based world-model RL.
7. DiT/DDPM on CIFAR-10 for diffusion-transformer generation.
8. Atari benchmark notebook to evaluate trained checkpoints.

Most training cells are commented out by default so documentation builds and first-time notebook opens do not launch expensive downloads or training jobs. Uncomment the final training cells after verifying optional dependencies and dataset paths.
