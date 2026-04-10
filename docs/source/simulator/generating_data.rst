Generating Training Data
======================

TorchWM can generate large datasets in various formats for training world models and RL agents.

Export Formats
-------------

HDF5 (Recommended for Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stores episodes efficiently for fast loading:

.. code-block:: bash

    python -m torchwm.sim.cli generate --episodes 1000 --steps 100 --out data/train.h5 --mode hdf5

Output structure:

.. code-block:: text

    train.h5/
    ├── episode_0/
    │   ├── frames    # [T, H, W, C] uint8
    │   ├── actions   # [T, ...] float32
    │   ├── rewards   # [T] float32
    │   ├── dones     # [T] uint8
    │   └── metadata  # JSON
    ├── episode_1/
    ...

Images + JSON
~~~~~~~~~~~~

Human-readable format for inspection:

.. code-block:: bash

    python -m torchwm.sim.cli generate --episodes 10 --steps 50 --out data/images --mode image_json

Output structure:

.. code-block:: text

    data/images/
    ├── episode_0/
    │   ├── frame_000000.png
    │   ├── frame_000001.png
    │   └── metadata.json
    ...

Python Generation
-----------------

Custom Data Collection:

.. code-block:: python

    from torchwm.sim.envs.basic_env import BasicEnv
    from torchwm.sim.exporters.hdf5 import HDF5Exporter

    config = {
        "physics": {"timestep": 1/60, "substeps": 1, "num_solver_iterations": 50, "gravity_z": -9.81},
        "generator": {
            "objects": [
                {"shape": {"type": "box", "size": [0.5, 0.5, 0.5]}, "position": [0, 0, 1], "mass": 1.0},
            ]
        },
        "camera": {"width": 64, "height": 64, "fov": 60, "position": [1, 1, 1], "target": [0, 0, 0]},
    }

    env = BasicEnv(config)

    with HDF5Exporter("training_data.h5", mode="w") as exp:
        for ep in range(1000):
            obs, info = env.reset(seed=ep)
            
            frames = [obs]
            actions = []
            rewards = []
            dones = []
            
            for t in range(100):
                # Random or policy actions
                action = [0.0, 0.0, 0.0]  # torque values
                
                obs, reward, done, info = env.step({
                    "body": 0,
                    "action": action,
                    "mode": "torque"
                })
                
                frames.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
            
            exp.add_episode(
                frames=frames,
                actions=actions,
                rewards=rewards,
                dones=dones,
                metadata={"seed": ep}
            )

    env.close()

With Random Object Placement:

.. code-block:: python

    config = {
        "physics": {"timestep": 1/60, "substeps": 1, "num_solver_iterations": 50, "gravity_z": -9.81},
        "generator": {
            "objects": [
                {
                    "shape": {"type": "box", "size": [0.5, 0.5, 0.5]},
                    "position_mean": [0, 0, 1],
                    "position_jitter": [0.5, 0.5, 0],
                    "random_orientation": True,
                    "mass": 1.0,
                }
            ]
        },
        "camera": {"width": 64, "height": 64, "fov": 60, "position": [1, 1, 1], "target": [0, 0, 0]},
    }

Multi-Worker Generation
----------------------

For faster large-scale generation:

.. code-block:: python

    from torchwm.sim.worker import MultiWorkerGenerator
    from torchwm.sim.exporters.hdf5 import HDF5Exporter

    config = {...}

    with HDF5Exporter("large_dataset.h5", mode="w") as exp:
        with MultiWorkerGenerator(config, num_workers=4, master_seed=42) as gen:
            gen.generate(
                num_episodes=1000,
                steps_per_episode=200,
                exporter=exp
            )

Loading Generated Data
---------------------

HDF5 can be loaded directly:

.. code-block:: python

    import h5py

    with h5py.File("training_data.h5", "r") as f:
        for ep_key in f.keys():
            ep = f[ep_key]
            frames = ep["frames"][:]        # shape: (T, H, W, C)
            actions = ep["actions"][:]
            rewards = ep["rewards"][:]
            dones = ep["dones"][:]
