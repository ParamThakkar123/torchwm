API Reference
=============

This reference is generated from source docstrings and grouped by workflow. Use
:doc:`world_models_guide` for conceptual explanations and this page for exact
classes, functions, and module-level APIs.

Public package surface
----------------------

These modules expose the most common imports and lazy constructors.

**Primary module:** ``torchwm``. Implementation modules are documented below for API completeness.

Use ``torchwm`` for common workflows::

   import torchwm
   agent = torchwm.create_model("dreamer", env="walker-walk")

.. automodule:: torchwm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.api
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

**Key classes:** ``Dreamer``, ``DreamerAgent``, ``RSSM``, ``RecurrentStateSpaceModel``, ``Planet``, ``ModularRSSM``, ``JEPAAgent``, ``VisionTransformer``, ``IRISAgent``, ``IRISTransformer``, ``IRISWorldModel``, ``Genie``, ``LatentActionModel``, and ``DynamicsModel``.

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

**Key classes:** ``DDPM``, ``DiT``, ``DiffusionUNet``, ``EDMPreconditioner``, ``EulerSampler``, ``RewardTerminationModel``, and ``ActorCriticNetwork``.

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

**Key classes:** ``ConvEncoder``, ``ConvDecoder``, ``DenseDecoder``, ``ActionDecoder``, ``CNNEncoder``, ``CNNDecoder``, ``IRISEncoder``, ``IRISDecoder``, ``DiscreteAutoencoder``, ``VectorQuantizer``, ``VectorQuantizerEMA``, ``VideoTokenizer``, ``MultiHeadSelfAttention``, and ``STTransformer``.

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

Environment adapters
~~~~~~~~~~~~~~~~~~~~

The environment APIs below mirror the dedicated environment guide pages: DMC,
Gym/Gymnasium, Atari/ALE, MuJoCo, Unity ML-Agents, and vectorization utilities.
DIAMOND-style Atari support is intentionally not listed as an environment
adapter because it is Atari preprocessing rather than a separate environment
family.

.. automodule:: world_models.envs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.dmc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.gym_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.ale_atari_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.ale_atari_vector_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.mujoco_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.robotics_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.unity_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.vector_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: world_models.envs.wrappers
   :members:
   :undoc-members:
   :show-inheritance:

Atari preprocessing helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These helpers wrap Atari environments for specific training recipes. They are
not separate environment families.

.. automodule:: world_models.envs.diamond_atari
   :members:
   :undoc-members:
   :show-inheritance:

Datasets and transforms
~~~~~~~~~~~~~~~~~~~~~~~

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
