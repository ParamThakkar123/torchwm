TorchWM | World Models & Simulator
=================================

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item::

      .. div:: hero-text

         .. raw:: html

            <h1 class="hero-title">TorchWM</h1>
            <p class="hero-subtitle">World Models & Deterministic Simulation</p>

         TorchWM is a modular PyTorch library for world models, latent-dynamics planning, and representation learning. It includes a deterministic PyBullet-based simulator for generating training data, training RL agents, and creating environments for world model research.

         .. grid:: 1 1 2 2
            :gutter: 2

            .. grid-item::

               .. button-ref:: getting_started
                  :ref-type: doc
                  :color: primary
                  :expand:

                  Get Started →

            .. grid-item::

               .. button-ref:: simulator
                  :ref-type: doc
                  :color: secondary
                  :expand:

                  Simulator Docs →

   .. grid-item::

      .. card:: Key Features
         :class: feature-card

         * **World Models** - Dreamer, JEPA, IRIS, DiT implementations
         * **Deterministic Sim** - Reproducible physics simulation
         * **RL Training** - Gymnasium, Stable-Baselines3, TorchRL
         * **Multi-Worker** - Parallel data generation
         * **Easy Integration** - Connect your existing models

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
