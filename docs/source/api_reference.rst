API Reference
=============

This reference is generated from source docstrings and grouped by subsystem.

Top-Level Package
-----------------

.. automodule:: world_models
   :members:
   :undoc-members:
   :show-inheritance:

Core Public APIs
----------------

.. automodule:: world_models.models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.configs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs
   :members:
   :undoc-members:
   :show-inheritance:

Transformer Blocks
-------------------

.. automodule:: world_models.blocks
   :members:
   :undoc-members:
   :show-inheritance:

Dreamer
-------

.. automodule:: world_models.models.dreamer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.dreamer_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.vision.dreamer_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.vision.dreamer_decoder
   :members:
   :undoc-members:
   :show-inheritance:

JEPA and ViT
------------

.. automodule:: world_models.models.jepa_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.training.train_jepa
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.vit
   :members:
   :undoc-members:
   :show-inheritance:

Masking Strategies
------------------

.. automodule:: world_models.masks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.masks.default
   :members:
   :undoc-members:
   :show-inheritance:

IRIS (Sample-Efficient World Models)
-------------------------------------

IRIS implements "Transformers are Sample-Efficient World Models" - a method that achieves
human-level performance on Atari with only 100k environment steps (~2 hours of gameplay)
by learning entirely in the imagination of a world model.

Architecture:
- Discrete autoencoder (VQVAE) compresses frames to tokens
- Autoregressive Transformer models dynamics
- Actor-Critic trains entirely in imagined trajectories

.. automodule:: world_models.configs.iris_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.iris_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.iris_transformer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.vision.iris_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.vision.iris_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.vision.vq_layer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.memory.iris_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.controller.iris_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.training.train_iris
   :members:
   :undoc-members:
   :show-inheritance:

Benchmarks
----------

.. automodule:: benchmarks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.benchmarks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.benchmarks.runner
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.benchmarks.adapters
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.benchmarks.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Diffusion
---------

.. automodule:: world_models.models.diffusion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.diffusion.DDPM
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.diffusion.DiT
   :members:
   :undoc-members:
   :show-inheritance:

Datasets and Transforms
-----------------------

.. automodule:: world_models.datasets.cifar10
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.datasets.imagenet1k
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.transforms.transforms
   :members:
   :undoc-members:
   :show-inheritance:

Memory and Controllers
----------------------

.. automodule:: world_models.memory.dreamer_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.memory.planet_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.controller.rssm_policy
   :members:
   :undoc-members:
   :show-inheritance:

Inference Operators
--------------------

.. automodule:: world_models.inference
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.inference.operators
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Utilities
---------

.. automodule:: world_models.utils.dreamer_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.utils.jepa_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.utils.utils
   :members:
   :undoc-members:
   :show-inheritance: