Integration with Existing World Models
====================================

TorchWM integrates easily with existing world model implementations like those in your library.

Using Your GymImageEnv
---------------------

Your library already has a ``GymImageEnv`` wrapper. Here's how to use it with TorchWM:

.. code-block:: python

    from world_models.envs.gym_env import GymImageEnv
    from torchwm.sim.envs.basic_env import BasicEnv
    from torchwm.sim.gym_wrapper import GymWrapperEnv

    # Create PyBullet environment
    config = {
        "physics": {"timestep": 1/60, "substeps": 1, "num_solver_iterations": 50, "gravity_z": -9.81},
        "generator": {"objects": [{"shape": {"type": "box", "size": [0.5, 0.5, 0.5]}, "position": [0, 0, 1], "mass": 1.0}]},
        "camera": {"width": 64, "height": 64, "fov": 60, "position": [1, 1, 1], "target": [0, 0, 0]},
    }

    basic_env = BasicEnv(config)
    gym_env = GymWrapperEnv(basic_env, action_config={"type": "torque", "body_index": 0})

    # Wrap with your GymImageEnv
    env = GymImageEnv(gym_env, size=(64, 64))

    # Now use with your world models!
    obs = env.reset()  # returns {"image": (3, 64, 64) uint8}
    obs, reward, done, info = env.step(env.action_space.sample())

Integration with Dreamer
-----------------------

.. code-block:: python

    from world_models.envs.gym_env import GymImageEnv
    from world_models.models.dreamer import Dreamer
    from torchwm.sim.envs.basic_env import BasicEnv
    from torchwm.sim.gym_wrapper import GymWrapperEnv

    # Create environment
    config = {...}
    env = GymImageEnv(
        GymWrapperEnv(BasicEnv(config), action_config={"type": "torque"}),
        size=(64, 64)
    )

    # Create Dreamer agent
    agent = Dreamer(
        obs_space=env.observation_space,
        action_space=env.action_space
    )

    # Training loop
    for episode in range(1000):
        obs = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            agent.update(obs, action, reward, done)

Integration with JEPA
-------------------

.. code-block:: python

    from world_models.envs.gym_env import GymImageEnv
    from world_models.models.jepa_agent import JEPAAgent
    from torchwm.sim.envs.basic_env import BasicEnv
    from torchwm.sim.gym_wrapper import GymWrapperEnv

    config = {...}
    env = GymImageEnv(
        GymWrapperEnv(BasicEnv(config), action_config={"type": "torque"}),
        size=(64, 64)
    )

    agent = JEPAAgent(obs_dim=(3, 64, 64), action_dim=3)

    # Collect trajectories
    for _ in range(100):
        obs = env.reset()
        for _ in range(100):
            action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            agent.store(obs, action, reward, done)

Register as Gym Environment
--------------------------

Register TorchWM as a gym environment ID:

.. code-block:: python

    import gym
    from gym import envs
    from world_models.envs.gym_env import GymImageEnv
    from torchwm.sim.envs.basic_env import BasicEnv
    from torchwm.sim.gym_wrapper import GymWrapperEnv

    class TorchWMEnv:
        def __init__(self, config=None):
            config = config or {
                "physics": {"timestep": 1/60, "substeps": 1, "num_solver_iterations": 50, "gravity_z": -9.81},
                "generator": {"objects": [{"shape": {"type": "box", "size": [0.5, 0.5, 0.5]}, "position": [0, 0, 1], "mass": 1.0}]},
                "camera": {"width": 64, "height": 64, "fov": 60, "position": [1, 1, 1], "target": [0, 0, 0]},
            }
            self._env = GymImageEnv(
                GymWrapperEnv(BasicEnv(config), action_config={"type": "torque", "body_index": 0}),
                size=(64, 64)
            )
        
        def __getattr__(self, name):
            return getattr(self._env, name)

    # Register
    envs.register(id="TorchWM-Box-v0", entry_point=TorchWMEnv)

    # Now use string ID
    env = gym.make("TorchWM-Box-v0")

Using Your Dataset Classes
------------------------

Load TorchWM data with your existing dataset classes:

.. code-block:: python

    from world_models.datasets.diamond_dataset import DiamondDataset

    # TorchWM generates standard HDF5 format
    # Just point your dataset to the file
    dataset = DiamondDataset(
        "path/to/torchwm_data.h5",
        image_size=64,
        sequence_length=16
    )
