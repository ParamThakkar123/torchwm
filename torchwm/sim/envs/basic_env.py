"""A simple BaseEnv implementation that composes a generator, PyBullet
adapter and CameraSensor. This env is intended as a minimal, deterministic
environment suitable for world-model dataset generation and testing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..api import BaseEnv, RNGManager, RNGStreams
from ..physics.pybullet_adapter import PyBulletAdapter
from ..sensors.camera import CameraSensor


class BasicEnv(BaseEnv):
    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self._physics = PyBulletAdapter()
        self._camera: Optional[CameraSensor] = None
        self._rng_manager: Optional[RNGManager] = None
        # store active RNGStreams by scope so we can snapshot/restore their state
        self._rngs: Dict[str, RNGStreams] = {}
        self._step_count = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        # Construct RNGManager from seed and derive scopes for generator, physics and sensors
        if seed is None:
            seed = 0
        self._rng_manager = RNGManager(seed)

        # Init physics with config and a physics-specific RNG (store stream)
        phys_rng = self._rng_manager.split("physics")
        self._rngs["physics"] = phys_rng
        phys_cfg = self.config.get("physics", {})
        self._physics.init_world(phys_cfg, phys_rng)

        # Spawn objects using generator config (simple inline generator for now)
        # Generator should consume rng from rng_manager.split("generator/<i>")
        gen = self.config.get("generator", {})
        objects = gen.get("objects", [])
        for i, proto in enumerate(objects):
            scope = f"generator/object_{i}"
            obj_rng = self._rng_manager.split(scope)
            self._rngs[scope] = obj_rng
            self._physics.spawn_object(proto, obj_rng)

        # Create camera sensor with dedicated rng and store it
        cam_cfg = self.config.get("camera", {})
        cam_scope = "sensors/camera"
        cam_rng = self._rng_manager.split(cam_scope)
        self._rngs[cam_scope] = cam_rng
        self._camera = CameraSensor(cam_cfg, self._physics, cam_rng)

        self._step_count = 0
        obs = self._camera.read() if self._camera is not None else None
        return obs, {"seed": seed}

    def step(self, action: Any):
        # Apply actions if provided. We expect actions to be a dict with
        # keys 'body': body_id, 'mode': 'torque'|'position', 'action': [...]
        if isinstance(action, dict) and action:
            body = action.get("body")
            act = action.get("action")
            mode = action.get("mode", "torque")
            if body is not None and act is not None:
                try:
                    self._physics.apply_action(body, act, mode=mode)
                except Exception:
                    # best-effort: ignore action failures in the simple env
                    pass

        # advance physics timestep
        dt = float(self.config.get("physics", {}).get("timestep", 1.0 / 60.0))
        self._physics.step(dt)
        self._step_count += 1
        obs = self._camera.read() if self._camera is not None else None
        reward = 0.0
        done = False
        info = {"step": self._step_count}
        return obs, reward, done, info

    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot that includes physics state and serialized RNG states."""
        phys_state = self._physics.get_state()
        rng_states = {}
        for k, stream in self._rngs.items():
            try:
                rng_states[k] = stream.to_state()
            except Exception:
                rng_states[k] = None

        return {"physics": phys_state, "rngs": rng_states}

    def restore(self, snapshot: Any) -> None:
        """Restore world and RNGs from a snapshot produced by `snapshot()`.

        Restores RNGStreams before restoring physics so any generator sampling
        that happens during spawn is deterministic.
        """
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot must be a dict returned by snapshot()")

        rng_states = snapshot.get("rngs", {}) or {}
        # reconstruct RNGStreams from saved states where possible
        for k, s in rng_states.items():
            if s is None:
                continue
            try:
                self._rngs[k] = RNGStreams.from_state(s)
            except Exception:
                # best-effort: ignore failures and continue
                pass

        # If we have a physics RNG restored, pass it to physics restore if needed
        # Now restore physics state (which re-spawns objects)
        phys_state = snapshot.get("physics")
        if phys_state is not None:
            self._physics.set_state(phys_state)

        # Recreate camera sensor with restored RNG if available
        cam_cfg = self.config.get("camera", {})
        cam_rng = self._rngs.get("sensors/camera")
        self._camera = CameraSensor(cam_cfg, self._physics, cam_rng)

    def render(self, mode: str = "rgb_array"):
        if self._camera is None:
            return None
        return self._camera.read()

    # (old snapshot/restore removed — new snapshot/restore implemented above)

    def close(self) -> None:
        self._physics.close()
