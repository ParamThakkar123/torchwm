Quick Start
===========

This guide will help you get started with the TorchWM simulator in minutes.

Installation
------------

First, ensure you have the required dependencies:

.. code-block:: bash

    pip install pybullet numpy pillow h5py

Generate Your First Environment
------------------------------

Generate a simple falling box environment:

.. code-block:: bash

    # Generate image dataset
    python -m torchwm.sim.cli generate --episodes 5 --steps 50 --out data/images --mode image_json

    # Generate HDF5 dataset
    python -m torchwm.sim.cli generate --episodes 10 --steps 100 --out data/sim.h5

What Just Happened?
-------------------

The simulator:

1. Created a PyBullet physics world with gravity
2. Spawned a box object at position (0, 0, 1)
3. Captured camera images at each timestep as the box falls
4. Exported frames to your chosen format

View Generated Data
-------------------

Check the output:

.. code-block:: bash

    ls data/images/
    # episode_0/ episode_1/ ...

    ls data/images/episode_0/
    # frame_000000.png ... metadata.json

Basic Python Usage
------------------

Create a custom environment:

.. code-block:: python

    from torchwm.sim.envs.basic_env import BasicEnv

    config = {
        "physics": {
            "timestep": 1/60,
            "substeps": 1,
            "num_solver_iterations": 50,
            "gravity_z": -9.81,
        },
        "generator": {
            "objects": [
                {
                    "shape": {"type": "box", "size": [0.5, 0.5, 0.5]},
                    "position": [0, 0, 1],
                    "mass": 1.0,
                }
            ]
        },
        "camera": {
            "width": 64,
            "height": 64,
            "fov": 60,
            "position": [1, 1, 1],
            "target": [0, 0, 0],
        },
    }

    env = BasicEnv(config)
    obs, info = env.reset(seed=0)

    # Step through the environment
    for _ in range(100):
        obs, reward, done, info = env.step({})

    env.close()

Next Steps
----------

* :doc:`configuration` - Learn how to customize environments
* :doc:`generating_data` - Generate datasets for training
* :doc:`rl_training` - Train RL agents
* :doc:`integration` - Integrate with your existing world models
