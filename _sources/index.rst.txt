TorchWM Documentation
======================

TorchWM is a modular PyTorch library for world models, latent-dynamics planning,
and representation learning. Train Dreamer, JEPA, IRIS, DiT, Genie, and DIAMOND
agents with a unified API.

.. code-block:: python

   import torchwm

   agent = torchwm.create_model("dreamer", action_size=6)
   agent.train(env_name="walker-walk", total_steps=100_000)

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   getting_started
   installation

.. toctree::
   :maxdepth: 1
   :caption: User Guides

   public_api
   operators_guide
   training_guide
   inference_guide
   evaluation_guide
   memory_guide
   environments_guide
   environments/index
   datasets/nuplan
   cli
   package_overview
   world_models_guide
   tutorials/world_model_env_rl_libraries
   modular_rssm_guide
   vision_guide
   datasets_guide
   losses_guide
   plugin_registry
   world_models_deep_dive

.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   dreamer
   jepa
   iris
   dit
   diamond
   genie

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api_reference
   configs_reference
   export_guide

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   benchmarks
