RNG contract and determinism guidelines
=====================================

This document describes the RNG contract for the torchwm simulator and how
determinism is achieved across components. A reproducible simulation run must
produce identical observations, metadata and exported data given the same
scenario configuration and top-level seed.

Top-level seed
--------------

All runs start from a single integer `seed` passed to environment creation or
reset. This seed is the only required input to reproduce a run; implementations
derive all further RNG streams from it deterministically.

Splitting
---------

Use a cryptographic hash (e.g. SHA256) of the top-level seed and a scope
string to derive independent integers for subcomponents. Example:

    scoped_seed = int(sha256(f"{seed}:{scope}").hexdigest()[:16], 16)

Then create the per-component RNG objects from `scoped_seed`:

- python.random: seed the global Random with scoped_seed
- numpy: create numpy.random.default_rng(scoped_seed)
- torch: create torch.Generator().manual_seed(scoped_seed)
- backend_seed: pass scoped_seed (or a derived int) to the physics
  backend when required

Name your scopes carefully ("generator", "sensors/camera", "physics/init",
"physics/contacts") so future changes preserve relative independence.

RNGStreams
----------

Each call to RNGManager.split(name) returns an RNGStreams container with
objects appropriate to the runtime. Store these RNGStreams alongside objects
that rely on them (e.g. the camera sensor keeps its RNGStreams for noise
sampling and augmentation).

Physics determinism (PyBullet)
------------------------------

To ensure reproducible stepping in PyBullet:

- Run PyBullet in DIRECT mode (no GUI).
- Use a fixed timestep and fixed number of internal substeps.
- Set solver iterations and other solver parameters explicitly.
- Avoid engine-level randomness for object placement: compute placement and
  orientation using the RNGStreams and then call PyBullet's spawn methods.

Snapshotting
------------

Snapshots must capture:

- All relevant physics state (positions, velocities, joint states).
- Per-object properties needed to re-spawn objects deterministically.
- The RNGStreams state or enough data to re-seed RNGs deterministically.

When restoring a snapshot, re-seed RNGs using the stored streams before
restoring physics state to ensure subsequent sampling is deterministic.

Versioning
----------

Store a simulator version string and the adapter version in exported datasets
and snapshots. If generators change, bump generator version and seed-scope
strings so old datasets remain reproducible.
