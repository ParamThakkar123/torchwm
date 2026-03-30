RNG and Determinism
==================

TorchWM is designed to be fully deterministic. This document explains how the RNG system works and how to ensure reproducible results.

RNG Architecture
----------------

The simulator uses a hierarchical RNG system:

1. **Master Seed**: Single integer passed to ``env.reset(seed=X)``
2. **RNGManager**: Derives independent streams for different components
3. **RNGStreams**: Container with separate RNGs for each subsystem

.. image:: /_images/rng_architecture.png
   :alt: RNG Architecture

How It Works
------------

When you call ``env.reset(seed=42)``:

1. An ``RNGManager(42)`` is created
2. It splits the seed into independent streams:

   - ``physics`` - for physics initialization
   - ``generator/object_0`` - for first object placement
   - ``generator/object_1`` - for second object placement
   - ``sensors/camera`` - for camera noise/jitter
   - etc.

3. Each stream is independent and deterministic

Seeding Example
---------------

.. code-block:: python

    from torchwm.sim.envs.basic_env import BasicEnv

    config = {...}

    # Two environments with same seed produce identical results
    env1 = BasicEnv(config)
    env2 = BasicEnv(config)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    # obs1 and obs2 are byte-identical!

    # Different seed = different result
    obs3, _ = env1.reset(seed=456)  # different from obs1

Ensuring Determinism
--------------------

To ensure deterministic behavior:

1. **Use integer seeds**: Always pass integer seeds

   .. code-block:: python

       env.reset(seed=0)       # good
       env.reset(seed=None)    # may vary

2. **Use fixed timestep**: Don't vary physics timestep

   .. code-block:: python

       "physics": {"timestep": 1/60}  # good - fixed

3. **Disable real-time mode**: Always use DIRECT mode (default)

4. **Match Python environment**: Results may vary across Python/NumPy versions

Snapshot and Restore
--------------------

Snapshots include full RNG state for exact replay:

.. code-block:: python

    # Save state
    env = BasicEnv(config)
    env.reset(seed=0)
    for _ in range(50):
        env.step({})

    snapshot = env.snapshot()
    # Save to disk...

    # Restore state
    env2 = BasicEnv(config)
    env2.restore(snapshot)

    # Continue from exact same state
    for _ in range(10):
        env2.step({})

Testing Determinism
-------------------

Run the built-in tests:

.. code-block:: bash

    pytest tests/test_sim.py -v

Or manually verify:

.. code-block:: python

    import numpy as np
    from torchwm.sim.envs.basic_env import BasicEnv

    config = {...}

    def run_episode(seed):
        env = BasicEnv(config)
        obs, _ = env.reset(seed=seed)
        frames = [obs]
        for _ in range(10):
            obs, _, _, _ = env.step({})
            frames.append(obs)
        env.close()
        return np.stack(frames)

    # Should be identical
    f1 = run_episode(123)
    f2 = run_episode(123)
    assert np.array_equal(f1, f2)  # passes

    # Should be different
    f3 = run_episode(456)
    assert not np.array_equal(f1, f3)  # passes
