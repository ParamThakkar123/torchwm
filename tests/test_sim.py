"""CI tests for determinism and API compatibility.

These tests run quickly and are suitable for CI pipelines.
"""

import tempfile
import os
import numpy as np
import pytest

from torchwm.sim.envs.basic_env import BasicEnv
from torchwm.sim.exporters.hdf5 import HDF5Exporter


def make_test_config():
    return {
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
                },
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


def run_episode(seed, steps=8):
    cfg = make_test_config()
    env = BasicEnv(cfg)
    obs, _ = env.reset(seed=seed)
    frames = [obs]
    for _ in range(steps - 1):
        obs, _, _, _ = env.step({})
        frames.append(obs)
    env.close()
    return np.stack(frames)


def test_determinism_same_seed():
    """Same seed should produce identical frames."""
    f1 = run_episode(123, 8)
    f2 = run_episode(123, 8)
    assert f1.shape == f2.shape
    assert np.array_equal(f1, f2)


def test_determinism_different_seeds():
    """Different seeds should produce different frames."""
    f1 = run_episode(123, 8)
    f2 = run_episode(456, 8)
    assert not np.array_equal(f1, f2)


def test_snapshot_restore():
    """Snapshot/restore should reproduce continuation."""
    cfg = make_test_config()
    env = BasicEnv(cfg)
    env.reset(seed=222)

    # Run 4 steps
    for _ in range(4):
        env.step({})

    snap = env.snapshot()

    # Continue 4 more steps
    frames_a = []
    for _ in range(4):
        obs, _, _, _ = env.step({})
        frames_a.append(obs)

    env.close()

    # Restore and continue
    env2 = BasicEnv(cfg)
    env2.restore(snap)
    frames_b = []
    for _ in range(4):
        obs, _, _, _ = env2.step({})
        frames_b.append(obs)
    env2.close()

    assert np.array_equal(np.stack(frames_a), np.stack(frames_b))


def test_export_import():
    """Export to HDF5 and verify contents."""
    cfg = make_test_config()
    env = BasicEnv(cfg)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name

    try:
        exporter = HDF5Exporter(path, mode="w")
        obs, _ = env.reset(seed=0)
        frames = [obs]
        for _ in range(5):
            obs, _, _, _ = env.step({})
            frames.append(obs)
        exporter.add_episode(
            frames=frames,
            actions=[[]] * 6,
            rewards=[0.0] * 6,
            dones=[False] * 6,
            metadata={"seed": 0},
        )
        exporter.close()
        env.close()

        # Verify file exists and has data
        import h5py

        with h5py.File(path, "r") as f:
            assert "episode_0" in f
            assert "frames" in f["episode_0"]
            assert f["episode_0"]["frames"].shape[0] == 6
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
