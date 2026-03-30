TorchWM Simulator
=================

TorchWM includes a deterministic PyBullet-based simulator for generating training data, training RL agents, and creating environments for world model research.

.. toctree::
   :maxdepth: 2

   simulator/quickstart
   simulator/configuration
   simulator/rng_determinism
   simulator/generating_data
   simulator/rl_training
   simulator/world_model_training
   simulator/integration

Key Features
-----------

* **Deterministic**: Full control over randomness for reproducible results
* **PyBullet Backend**: Accurate physics simulation
* **Multiple Export Formats**: HDF5, images+JSON, TFRecord
* **Gymnasium Compatible**: Works with Stable-Baselines3, TorchRL, and more
* **World Model Ready**: Generate datasets for Dreamer, JEPA, IRIS, etc.
* **Multi-Worker**: Parallel episode generation with deterministic seeding

Quick Links
----------

* :doc:`simulator/quickstart` - Get started in 5 minutes
* :doc:`simulator/configuration` - Customize environments
* :doc:`simulator/integration` - Connect to your existing world models
