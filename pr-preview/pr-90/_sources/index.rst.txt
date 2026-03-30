TorchWM | World Models & Simulator
=================================

**TorchWM** is a modular PyTorch library for world models, latent-dynamics planning, and representation learning. It includes a deterministic PyBullet-based simulator for generating training data, training RL agents, and creating environments for world model research.

Key Features
------------

* **World Models** - Dreamer, JEPA, IRIS, DiT implementations
* **Deterministic Sim** - Reproducible physics simulation
* **RL Training** - Gymnasium, Stable-Baselines3, TorchRL  
* **Multi-Worker** - Parallel data generation
* **Easy Integration** - Connect your existing models

Quick Links
----------

* :doc:`getting_started` - Get started with TorchWM
* :doc:`simulator` - Simulator documentation
* :doc:`dreamer` - Dreamer implementation
* :doc:`jepa` - JEPA implementation
* :doc:`iris` - IRIS implementation
* :doc:`dit` - DiT implementation

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   getting_started
   package_overview

.. toctree::
   :maxdepth: 2
   :caption: Simulator
   :hidden:

   simulator

.. toctree::
   :maxdepth: 2
   :caption: Algorithms
   :hidden:

   dreamer
   jepa
   iris
   dit

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api_reference
