TorchWM Simulator
================

TorchWM includes a deterministic PyBullet-based simulator for generating training data, training RL agents, and creating environments for world model research.

Key Features
-----------

* **Deterministic** - Full control over randomness for reproducible results
* **PyBullet Backend** - Accurate physics simulation
* **Multiple Export Formats** - HDF5, images+JSON, TFRecord
* **Gymnasium Compatible** - Works with Stable-Baselines3, TorchRL
* **World Model Ready** - Generate datasets for Dreamer, JEPA, IRIS
* **Multi-Worker** - Parallel episode generation with deterministic seeding

Quick Start
-----------

Generate your first environment in seconds:

.. code-block:: bash

    # Generate training data
    python -m torchwm.sim.cli generate --episodes 100 --steps 100 --out data.h5

    # Or use Python
    from torchwm.sim.envs.basic_env import BasicEnv
    
    config = {...}
    env = BasicEnv(config)
    obs, info = env.reset(seed=0)

Documentation
------------

.. toctree::
   :maxdepth: 2

   simulator/quickstart
   simulator/configuration
   simulator/rng_determinism
   simulator/generating_data
   simulator/rl_training
   simulator/world_model_training
   simulator/integration
