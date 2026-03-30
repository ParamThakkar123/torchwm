"""PyBullet physics adapter implementing the PhysicsAdapter interface.

This adapter runs PyBullet in DIRECT mode and provides deterministic,
fixed-timestep stepping, spawn/load/save (serializable) snapshot and restore
functionality. It purposely avoids engine-level randomness by using the
provided RNGStreams for object placement and any sampling.

Notes:
- The adapter stores per-object `proto` dictionaries supplied to
  `spawn_object` so snapshots can re-create the world deterministically.
- Snapshot data is a plain Python dict that can be serialized (JSON-friendly
  where possible). Binary-heavy assets (URDF files) are referenced by path in
  the proto and must be available when restoring.
"""

from __future__ import annotations

import math
import json
from typing import Any, Dict, Optional, Sequence, List

try:
    import pybullet as p
    import pybullet_data
except Exception:  # pragma: no cover - tests may not have pybullet
    p = None

import numpy as np

from ..api import PhysicsAdapter, RNGStreams


class PyBulletAdapter(PhysicsAdapter):
    """PyBullet adapter.

    Usage:
        adapter = PyBulletAdapter()
        adapter.init_world(config, rng)
        obj_id = adapter.spawn_object(proto, rng)
        adapter.step(dt)
        state = adapter.get_state()
        adapter.set_state(state)
    """

    def __init__(self) -> None:
        if p is None:
            raise ImportError("pybullet is required for PyBulletAdapter")
        self._cid: Optional[int] = None
        self._timestep: float = 1.0 / 60.0
        self._substeps: int = 1
        self._num_solver_iterations: int = 50
        # Keep list of spawned protos for deterministic re-spawn on restore
        self._spawned_protos: List[Dict[str, Any]] = []
        # Map from body unique id to proto index; body ids change across sessions
        self._body_uids: List[int] = []

    def init_world(
        self, config: Dict[str, Any], rng: Optional[RNGStreams] = None
    ) -> None:
        # Connect in DIRECT mode for headless deterministic runs
        if self._cid is not None:
            try:
                p.disconnect(self._cid)
            except Exception:
                pass

        self._cid = p.connect(p.DIRECT)

        # set deterministic physics parameters
        self._timestep = float(config.get("timestep", 1.0 / 60.0))
        self._substeps = int(config.get("substeps", 1))
        self._num_solver_iterations = int(config.get("num_solver_iterations", 50))

        # Apply common deterministic settings
        try:
            p.setPhysicsEngineParameter(
                numSolverIterations=self._num_solver_iterations,
                deterministicOverlappingPairs=True,
            )
        except TypeError:
            # Older pybullet may not accept deterministicOverlappingPairs
            try:
                p.setPhysicsEngineParameter(
                    numSolverIterations=self._num_solver_iterations
                )
            except Exception:
                pass

        p.setTimeStep(self._timestep)
        p.setGravity(0, 0, float(config.get("gravity_z", -9.81)))

        # Optionally set search path for URDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Reset simulation and clear tracked lists
        p.resetSimulation()
        self._spawned_protos = []
        self._body_uids = []

        # optionally use backend_seed from rng to control any backend-specific
        # determinism; pybullet itself doesn't provide a seed entrypoint for
        # physics RNG, so we rely on deterministic placement from the caller.

    def spawn_object(
        self, proto: Dict[str, Any], rng: Optional[RNGStreams] = None
    ) -> int:
        """Spawn an object described by `proto` and return PyBullet body id.

        Proto accepted keys (informal):
            - 'urdf': path to URDF file (preferred)
            - 'position': [x,y,z] float
            - 'orientation': [x,y,z,w] quaternion
            - 'fixed_base': bool
            - 'shape': {'type': 'box'|'sphere'|'capsule', 'size': [...]}
            - any other backend-specific kwargs
        """
        # Ensure deterministic placement: if position/orientation are absent,
        # derive them from rng (if provided) using numpy.RandomGenerator
        pos = proto.get("position")
        orn = proto.get("orientation")
        if pos is None:
            # sample positional jitter deterministically
            if rng is not None and getattr(rng, "numpy_rng", None) is not None:
                rng_np = rng.numpy_rng
            else:
                rng_np = np.random.default_rng(0)
            pos = proto.get("position_mean", [0.0, 0.0, 0.0])
            jitter = np.asarray(proto.get("position_jitter", [0.0, 0.0, 0.0]))
            pos = list(
                np.asarray(pos) + rng_np.uniform(-1.0, 1.0, size=3) * np.asarray(jitter)
            )

        if orn is None:
            if rng is not None and getattr(rng, "numpy_rng", None) is not None:
                rng_np = rng.numpy_rng
            else:
                rng_np = np.random.default_rng(0)
            # default: no rotation
            if proto.get("random_orientation", False):
                # sample a random unit quaternion
                u1 = float(rng_np.random())
                u2 = float(rng_np.random())
                u3 = float(rng_np.random())
                q = [
                    math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
                    math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
                    math.sqrt(u1) * math.sin(2 * math.pi * u3),
                    math.sqrt(u1) * math.cos(2 * math.pi * u3),
                ]
                orn = q
            else:
                orn = proto.get("orientation", [0, 0, 0, 1])

        # Spawn via URDF if provided
        fixed_base = bool(proto.get("fixed_base", False))
        body_id: int
        if "urdf" in proto:
            urdf_path = proto["urdf"]
            flags = proto.get("flags", 0)
            body_id = p.loadURDF(
                urdf_path,
                basePosition=pos,
                baseOrientation=orn,
                useFixedBase=fixed_base,
                flags=flags,
            )
        elif "shape" in proto:
            shape = proto["shape"]
            stype = shape.get("type", "box")
            if stype == "box":
                half_extents = np.asarray(shape.get("size", [1.0, 1.0, 1.0])) / 2.0
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half_extents.tolist()
                )
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents.tolist())
                body_id = p.createMultiBody(
                    baseMass=float(proto.get("mass", 1.0)),
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=pos,
                    baseOrientation=orn,
                )
            elif stype == "sphere":
                r = float(shape.get("radius", 0.5))
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=r)
                body_id = p.createMultiBody(
                    baseMass=float(proto.get("mass", 1.0)),
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=pos,
                    baseOrientation=orn,
                )
            else:
                # Fallback: treat as box
                half_extents = np.asarray(shape.get("size", [1.0, 1.0, 1.0])) / 2.0
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half_extents.tolist()
                )
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents.tolist())
                body_id = p.createMultiBody(
                    baseMass=float(proto.get("mass", 1.0)),
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=pos,
                    baseOrientation=orn,
                )
        else:
            raise ValueError("proto must contain either 'urdf' or 'shape'")

        # Record proto and body mapping for snapshot/restore
        self._spawned_protos.append(dict(proto))
        self._body_uids.append(body_id)

        # Optionally set dynamics parameters if provided
        dyn = proto.get("dynamics", {})
        if dyn:
            for k, v in dyn.items():
                try:
                    # Common keys: lateralFriction, restitution, etc.
                    if k == "lateralFriction":
                        p.changeDynamics(body_id, -1, lateralFriction=float(v))
                    elif k == "restitution":
                        p.changeDynamics(body_id, -1, restitution=float(v))
                    else:
                        # best-effort: pass unknown keys to changeDynamics if signature matches
                        pass
                except Exception:
                    pass

        return body_id

    def step(self, dt: float) -> None:
        """Advance the world by dt seconds using fixed internal timestep.

        PyBullet is stepped in increments of the configured timestep; if dt is
        a multiple of the timestep we run the necessary number of steps.
        """
        if self._cid is None:
            raise RuntimeError("World not initialized. Call init_world first.")

        steps = max(1, int(round(dt / self._timestep)))
        for _ in range(steps):
            # perform substeps if configured (simple loop)
            for _ in range(self._substeps):
                p.stepSimulation()

    def get_state(self) -> Dict[str, Any]:
        """Return a serializable representation of the physics state.

        The returned dict contains per-object protos and their dynamic state
        so the caller can restore the world deterministically.
        """
        if self._cid is None:
            raise RuntimeError("World not initialized. Call init_world first.")

        state: Dict[str, Any] = {"bodies": []}
        num_bodies = p.getNumBodies()
        for uid in range(num_bodies):
            # We assume body indices correspond to spawn order in the current
            # process; attempt to find a proto for this body if available.
            try:
                pos, orn = p.getBasePositionAndOrientation(uid)
                linv, angv = p.getBaseVelocity(uid)
            except Exception:
                # If body was removed or query failed, skip
                continue

            # Gather joint states
            joints = []
            n_joints = p.getNumJoints(uid)
            for j in range(n_joints):
                try:
                    js = p.getJointState(uid, j)
                    # js returns (position, velocity, reactionForces, appliedTorque)
                    joints.append(
                        {
                            "position": js[0],
                            "velocity": js[1],
                        }
                    )
                except Exception:
                    joints.append({"position": None, "velocity": None})

            proto = None
            # try to map uid to our recorded protos; body ids may not match
            if uid < len(self._spawned_protos):
                proto = self._spawned_protos[uid]

            body_record = {
                "uid": int(uid),
                "proto": proto,
                "position": list(pos),
                "orientation": list(orn),
                "linear_velocity": list(linv),
                "angular_velocity": list(angv),
                "joints": joints,
            }
            state["bodies"].append(body_record)

        # Include PyBullet internal time step and parameters for completeness
        state["timestep"] = self._timestep
        state["substeps"] = self._substeps
        state["num_solver_iterations"] = self._num_solver_iterations

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore a world state previously produced by `get_state()`.

        This implementation resets the simulation and re-spawns protos in the
        same order they were recorded. It then sets base positions, velocities
        and joint states to match the recorded values.
        """
        if self._cid is None:
            # initialize default world if not yet initialized
            self.init_world(
                {
                    "timestep": state.get("timestep", self._timestep),
                    "substeps": state.get("substeps", self._substeps),
                    "num_solver_iterations": state.get(
                        "num_solver_iterations", self._num_solver_iterations
                    ),
                },
                None,
            )

        # Reset simulation and re-init lists
        p.resetSimulation()
        p.setTimeStep(state.get("timestep", self._timestep))
        self._spawned_protos = []
        self._body_uids = []

        # Recreate bodies from protos recorded in state; if proto is None,
        # attempt to create a placeholder box with recorded dims.
        for b in state.get("bodies", []):
            proto = b.get("proto") or {}
            # If the proto had no explicit position, pass recorded position
            proto_copy = dict(proto)
            proto_copy["position"] = b.get("position")
            proto_copy["orientation"] = b.get("orientation")
            body_id = self.spawn_object(proto_copy, None)

            # Set velocities
            linv = b.get("linear_velocity", [0.0, 0.0, 0.0])
            angv = b.get("angular_velocity", [0.0, 0.0, 0.0])
            try:
                p.resetBaseVelocity(body_id, linearVelocity=linv, angularVelocity=angv)
            except Exception:
                pass

            # Set joint states if present
            joints = b.get("joints", [])
            for j_idx, jrec in enumerate(joints):
                pos = jrec.get("position")
                vel = jrec.get("velocity")
                if pos is not None:
                    try:
                        p.resetJointState(
                            body_id, j_idx, targetValue=pos, targetVelocity=vel or 0.0
                        )
                    except Exception:
                        pass

        # Restore physics parameters if present
        self._timestep = state.get("timestep", self._timestep)
        self._substeps = state.get("substeps", self._substeps)
        self._num_solver_iterations = state.get(
            "num_solver_iterations", self._num_solver_iterations
        )
        try:
            p.setPhysicsEngineParameter(
                numSolverIterations=self._num_solver_iterations,
                deterministicOverlappingPairs=True,
            )
        except Exception:
            try:
                p.setPhysicsEngineParameter(
                    numSolverIterations=self._num_solver_iterations
                )
            except Exception:
                pass

    def apply_action(self, body_index: int, action: Any, mode: str = "torque") -> None:
        """Apply an action to a body. Supports simple torque or position control
        for base-level joints. `body_index` is the body unique id returned by
        spawn_object (PyBullet body id).
        """
        if self._cid is None:
            raise RuntimeError("World not initialized. Call init_world first.")

        if mode == "torque":
            # Expect action to be a list of per-joint torques
            torques = list(action)
            n_joints = p.getNumJoints(body_index)
            for j in range(min(n_joints, len(torques))):
                try:
                    p.setJointMotorControl2(
                        bodyIndex=body_index,
                        jointIndex=j,
                        controlMode=p.TORQUE_CONTROL,
                        force=float(torques[j]),
                    )
                except Exception:
                    pass
        elif mode == "position":
            # Expect action to be list of desired joint positions
            positions = list(action)
            n_joints = p.getNumJoints(body_index)
            for j in range(min(n_joints, len(positions))):
                try:
                    p.setJointMotorControl2(
                        bodyIndex=body_index,
                        jointIndex=j,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=float(positions[j]),
                    )
                except Exception:
                    pass
        else:
            raise ValueError(f"Unknown action mode: {mode}")

    def close(self) -> None:
        if self._cid is not None:
            try:
                p.disconnect(self._cid)
            except Exception:
                pass
            self._cid = None
