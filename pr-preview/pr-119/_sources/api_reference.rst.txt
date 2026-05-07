API Reference
=============

This reference is generated from source docstrings and grouped by subsystem.

Core Public APIs
----------------

.. automodule:: world_models.models
   :members:
   :show-inheritance:

.. automodule:: world_models.configs
   :members:
   :show-inheritance:

.. automodule:: world_models.envs
   :members:
   :show-inheritance:

Dreamer
-------

.. automodule:: world_models.models.dreamer
   :members:
   :show-inheritance:

.. automodule:: world_models.models.dreamer_rssm
   :members:
   :show-inheritance:

.. automodule:: world_models.vision.dreamer_encoder
   :members:
   :show-inheritance:

.. automodule:: world_models.vision.dreamer_decoder
   :members:
   :show-inheritance:

JEPA and ViT
------------

.. automodule:: world_models.models.jepa_agent
   :members:
   :show-inheritance:

.. automodule:: world_models.training.train_jepa
   :members:
   :show-inheritance:

.. automodule:: world_models.models.vit
   :members:
   :show-inheritance:

.. automodule:: world_models.masks.multiblock
   :members:
   :show-inheritance:

.. automodule:: world_models.masks.random
   :members:
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
   :show-inheritance:

.. automodule:: world_models.models.iris_agent
   :members:
   :show-inheritance:

.. automodule:: world_models.models.iris_transformer
   :members:
   :show-inheritance:

.. automodule:: world_models.vision.iris_encoder
   :members:
   :show-inheritance:

.. automodule:: world_models.vision.iris_decoder
   :members:
   :show-inheritance:

.. automodule:: world_models.vision.vq_layer
   :members:
   :show-inheritance:

.. automodule:: world_models.memory.iris_memory
   :members:
   :show-inheritance:

.. automodule:: world_models.controller.iris_policy
   :members:
   :show-inheritance:

.. automodule:: world_models.training.train_iris
   :members:
   :show-inheritance:

Benchmarks
^^^^^^^^^^

.. automodule:: benchmarks.atari_100k
   :members:
   :show-inheritance:

Diffusion
---------

.. automodule:: world_models.models.diffusion.DDPM
   :members:
   :show-inheritance:

.. automodule:: world_models.models.diffusion.DiT
   :members:
   :show-inheritance:

Datasets and Transforms
-----------------------

.. automodule:: world_models.datasets.cifar10
   :members:
   :show-inheritance:

.. automodule:: world_models.datasets.imagenet1k
   :members:
   :show-inheritance:

.. automodule:: world_models.transforms.transforms
   :members:
   :show-inheritance:

Memory and Controllers
----------------------

.. automodule:: world_models.memory.dreamer_memory
   :members:
   :show-inheritance:

.. automodule:: world_models.memory.planet_memory
   :members:
   :show-inheritance:

.. automodule:: world_models.controller.rssm_policy
   :members:
   :show-inheritance:

Utilities
---------

.. automodule:: world_models.utils.dreamer_utils
   :members:
   :show-inheritance:

.. automodule:: world_models.utils.jepa_utils
   :members:
   :show-inheritance:

.. automodule:: world_models.utils.utils
   :members:
   :show-inheritance:
