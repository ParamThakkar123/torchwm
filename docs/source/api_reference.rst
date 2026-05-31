API Reference
=============

This reference is generated from source docstrings and grouped by workflow. Use
:doc:`world_models_guide` for conceptual explanations and this page for exact
classes, functions, and module-level APIs.

Public package surface
----------------------

These modules expose the most common imports and lazy constructors.

.. autosummary::
   :toctree: generated
   :nosignatures:

   world_models
   world_models.models
   world_models.configs
   world_models.envs
   world_models.inference

.. automodule:: world_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models
   :members:
   :undoc-members:
   :show-inheritance:

Model catalog
-------------

Core model families
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   world_models.models.dreamer.Dreamer
   world_models.models.dreamer.DreamerAgent
   world_models.models.dreamer_rssm.RSSM
   world_models.models.rssm.RecurrentStateSpaceModel
   world_models.models.planet.Planet
   world_models.models.modular_rssm.ModularRSSM
   world_models.models.jepa_agent.JEPAAgent
   world_models.models.vit.VisionTransformer
   world_models.models.iris_agent.IRISAgent
   world_models.models.iris_transformer.IRISTransformer
   world_models.models.iris_transformer.IRISWorldModel
   world_models.models.genie.Genie
   world_models.models.latent_action_model.LatentActionModel
   world_models.models.dynamics_model.DynamicsModel

.. automodule:: world_models.models.dreamer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.dreamer_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.planet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.modular_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.jepa_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.vit
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

.. automodule:: world_models.models.genie
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.latent_action_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.dynamics_model
   :members:
   :undoc-members:
   :show-inheritance:

Diffusion and DIAMOND components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   world_models.models.diffusion.DDPM.DDPM
   world_models.models.diffusion.DiT.DiT
   world_models.models.diffusion.diamond_diffusion.DiffusionUNet
   world_models.models.diffusion.diamond_diffusion.EDMPreconditioner
   world_models.models.diffusion.diamond_diffusion.EulerSampler
   world_models.models.diffusion.reward_termination.RewardTerminationModel
   world_models.models.diffusion.actor_critic.ActorCriticNetwork

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

.. automodule:: world_models.models.diffusion.diamond_diffusion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.diffusion.reward_termination
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.models.diffusion.actor_critic
   :members:
   :undoc-members:
   :show-inheritance:

Vision, tokenization, and layers
--------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   world_models.vision.dreamer_encoder.ConvEncoder
   world_models.vision.dreamer_decoder.ConvDecoder
   world_models.vision.dreamer_decoder.DenseDecoder
   world_models.vision.dreamer_decoder.ActionDecoder
   world_models.vision.planet_encoder.CNNEncoder
   world_models.vision.planet_decoder.CNNDecoder
   world_models.vision.iris_encoder.IRISEncoder
   world_models.vision.iris_decoder.IRISDecoder
   world_models.vision.iris_decoder.DiscreteAutoencoder
   world_models.vision.vq_layer.VectorQuantizer
   world_models.vision.vq_layer.VectorQuantizerEMA
   world_models.vision.video_tokenizer.VideoTokenizer
   world_models.blocks.mhsa.MultiHeadSelfAttention
   world_models.blocks.st_transformer.STTransformer

.. automodule:: world_models.vision.dreamer_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.vision.dreamer_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.vision.planet_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.vision.planet_decoder
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

.. automodule:: world_models.vision.video_tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.blocks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.blocks.mhsa
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.blocks.st_transformer
   :members:
   :undoc-members:
   :show-inheritance:

Configuration objects
---------------------

.. automodule:: world_models.configs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.configs.dreamer_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.configs.jepa_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.configs.iris_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.configs.genie_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.configs.dit_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.configs.diamond_config
   :members:
   :undoc-members:
   :show-inheritance:

Training entry points
---------------------

.. automodule:: world_models.training.train_jepa
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.training.train_iris
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.training.train_genie
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.training.train_planet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.training.train_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.training.train_diamond
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.training.rl_harness
   :members:
   :undoc-members:
   :show-inheritance:

Memory, controllers, and inference operators
--------------------------------------------

.. automodule:: world_models.memory.dreamer_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.memory.planet_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.memory.iris_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.controller.rssm_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.controller.iris_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.controller.rollout_generator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.inference.operators
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: world_models.inference.operators.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.inference.operators.dreamer_operator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.inference.operators.planet_operator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.inference.operators.iris_operator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.inference.operators.jepa_operator
   :members:
   :undoc-members:
   :show-inheritance:

Datasets, environments, and transforms
--------------------------------------

.. automodule:: world_models.envs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.wrappers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.vector_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.datasets.video_datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.datasets.tinyworlds
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.datasets.diamond_dataset
   :members:
   :undoc-members:
   :show-inheritance:

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

Masking and JEPA helpers
------------------------

.. automodule:: world_models.masks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.masks.default
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.masks.multiblock
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.masks.random
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.helpers.jepa_helper
   :members:
   :undoc-members:
   :show-inheritance:

Benchmarks and reports
----------------------

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

.. automodule:: world_models.benchmarks.reporting
   :members:
   :undoc-members:
   :show-inheritance:

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

.. automodule:: world_models.utils.data_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.utils.jit_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.utils.memory_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.utils.logging_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.utils.utils
   :members:
   :undoc-members:
   :show-inheritance:
