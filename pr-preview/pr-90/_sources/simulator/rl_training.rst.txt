RL Training
==========

TorchWM provides Gymnasium-compatible wrappers for training RL agents.

Basic RL Training
----------------

Using Stable-Baselines3:

.. code-block:: python

    from torchwm.sim.envs.basic_env import BasicEnv
    from torchwm.sim.gym_wrapper import GymWrapperEnv
    from stable_baselines3 import PPO

    config = {
        "physics": {"timestep": 1/60, "substeps": 1, "num_solver_iterations": 50, "gravity_z": -9.81},
        "generator": {"objects": [{"shape": {"type": "box", "size": [0.5, 0.5, 0.5]}, "position": [0, 0, 1], "mass": 1.0}]},
        "camera": {"width": 64, "height": 64, "fov": 60, "position": [1, 1, 1], "target": [0, 0, 0]},
    }

    # Create environment
    env = BasicEnv(config)
    gym_env = GymWrapperEnv(
        env,
        sensors=["camera"],
        action_config={"type": "torque", "body_index": 0}
    )

    # Train PPO
    model = PPO("CnnPolicy", gym_env, verbose=1)
    model.learn(total_timesteps=100000)

    # Save model
    model.save("ppo_torchwm")

    # Inference
    obs, _ = gym_env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = gym_env.step(action)

    gym_env.close()

Vectorized Training
-------------------

For faster training with multiple environments:

.. code-block:: python

    from torchwm.sim.envs.basic_env import BasicEnv
    from torchwm.sim.gym_wrapper import GymWrapperEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        config = {...}
        env = BasicEnv(config)
        return GymWrapperEnv(env, action_config={"type": "torque"})

    env = DummyVecEnv([make_env for _ in range(4)])
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

Action Configuration
-------------------

Torque Control (default):

.. code-block:: python

    gym_env = GymWrapperEnv(
        env,
        action_config={
            "type": "torque",
            "body_index": 0,
        }
    )
    # Actions are joint torques

Position Control:

.. code-block:: python

    gym_env = GymWrapperEnv(
        env,
        action_config={
            "type": "position",
            "body_index": 0,
        }
    )
    # Actions are target joint positions

Custom Joints:

.. code-block:: python

    gym_env = GymWrapperEnv(
        env,
        action_config={
            "type": "torque",
            "body_index": 0,
            "joint_indices": [0, 1],  # Only control first 2 joints
            "low": -0.5,
            "high": 0.5,
        }
    )

Observation Wrappers
-------------------

Add frame stacking and normalization:

.. code-block:: python

    from torchwm.sim.wrappers.observation import FrameStackWrapper, NormalizeWrapper

    gym_env = GymWrapperEnv(env, action_config={...})
    gym_env = FrameStackWrapper(gym_env, num_stack=4)
    gym_env = NormalizeWrapper(gym_env, range_min=-1, range_max=1)
