"""Small example showing how to wrap BasicEnv with GymWrapperEnv and run
one training loop iteration (no learning, just env interaction).
"""

from __future__ import annotations

import time

import numpy as np

from torchwm.sim.envs.basic_env import BasicEnv
from torchwm.sim.gym_wrapper import GymWrapperEnv


def main():
    cfg = {
        "physics": {
            "timestep": 1.0 / 60.0,
            "substeps": 1,
            "num_solver_iterations": 50,
            "gravity_z": -9.81,
        },
        "generator": {
            "objects": [
                {
                    "shape": {"type": "box", "size": [0.5, 0.5, 0.5]},
                    "position": [0.0, 0.0, 1.0],
                    "mass": 1.0,
                }
            ]
        },
        "camera": {
            "width": 64,
            "height": 64,
            "fov": 60.0,
            "position": [1.0, 1.0, 1.0],
            "target": [0.0, 0.0, 0.0],
        },
    }

    env = BasicEnv(cfg)
    # Use action_config to specify torque control on first body
    gym_env = GymWrapperEnv(
        env, sensors=["camera"], action_config={"type": "torque", "body_index": 0}
    )

    obs, info = gym_env.reset(seed=123)
    print("Reset observation shape:", obs.shape)

    for t in range(10):
        action = np.zeros((gym_env.action_space.shape[0],), dtype=np.float32)
        obs, reward, done, info = gym_env.step(action)
        print(
            f"Step {t}: obs shape {None if obs is None else obs.shape}, reward={reward}"
        )
        time.sleep(0.01)

    gym_env.close()


if __name__ == "__main__":
    main()
