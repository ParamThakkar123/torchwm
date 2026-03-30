TorchWM Simulator
================

.. grid:: 1 1 2 2

   .. grid-item::

      .. card:: 🔬 Deterministic Simulation
         :class: shadow-sm
         
         Full control over randomness for reproducible results. Same seed = identical runs every time.

   .. grid-item::

      .. card:: 🎮 Multi-Purpose
         :class: shadow-sm
         
         RL training, world model generation, robotics simulation, and game environment creation.

   .. grid-item::

      .. card:: ⚡ Fast & Scalable
         :class: shadow-sm
         
         Multi-worker generation, batch processing, and optimized for large-scale datasets.

   .. grid-item::

      .. card:: 🔗 Easy Integration
         :class: shadow-sm
         
         Works with Stable-Baselines3, PyTorch, Dreamer, JEPA, and your existing world models.

Quick Start
-----------

Generate your first environment in seconds:

.. code-block:: bash

    # Generate training data
    python -m torchwm.sim.cli generate --episodes 100 --steps 100 --out data.h5

    # Or use Python
    from torchwm.sim.envs.basic_env import BasicEnv
    
    env = BasicEnv(config)
    obs, info = env.reset(seed=0)

.. toctree::
   :maxdepth: 1
   :hidden:

   simulator/quickstart
   simulator/configuration
   simulator/rng_determinism
   simulator/generating_data
   simulator/rl_training
   simulator/world_model_training
   simulator/integration
