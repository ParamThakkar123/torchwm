"""Deterministic tests for the simulator.

Two quick tests:
 - same_seed_produces_identical_frames
 - snapshot_restore_reproduces_continuation
"""

import tempfile
import numpy as np

from torchwm.sim.envs.basic_env import BasicEnv


def run_episode(seed: int, steps: int):
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
            "width": 32,
            "height": 32,
            "fov": 60.0,
            "position": [1.0, 1.0, 1.0],
            "target": [0.0, 0.0, 0.0],
        },
    }
    env = BasicEnv(cfg)
    obs, info = env.reset(seed=seed)
    frames = [obs]
    for t in range(steps - 1):
        obs, r, d, info = env.step({})
        frames.append(obs)
    env.close()
    return np.stack(frames)


def test_same_seed_identical_frames():
    f1 = run_episode(123, 8)
    f2 = run_episode(123, 8)
    assert f1.shape == f2.shape
    assert np.array_equal(f1, f2)


def test_snapshot_restore_continuation():
    env = BasicEnv(
        {
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
                "width": 32,
                "height": 32,
                "fov": 60.0,
                "position": [1.0, 1.0, 1.0],
                "target": [0.0, 0.0, 0.0],
            },
        }
    )
    obs, info = env.reset(seed=222)
    frames_a = [obs]
    for _ in range(4):
        obs, r, d, info = env.step({})
        frames_a.append(obs)

    snap = env.snapshot()

    # continue for more steps
    for _ in range(4):
        obs, r, d, info = env.step({})
        frames_a.append(obs)

    env.close()

    # restore into a new env and resume from snapshot
    env2 = BasicEnv(
        {
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
                "width": 32,
                "height": 32,
                "fov": 60.0,
                "position": [1.0, 1.0, 1.0],
                "target": [0.0, 0.0, 0.0],
            },
        }
    )
    env2.restore(snap)
    frames_b = []
    for _ in range(4):
        obs, r, d, info = env2.step({})
        frames_b.append(obs)
    env2.close()

    # frames_a continuation vs frames_b should match for the resumed portion
    cont_a = np.stack(frames_a[5:9])
    cont_b = np.stack(frames_b)
    assert cont_a.shape == cont_b.shape
    assert np.array_equal(cont_a, cont_b)
