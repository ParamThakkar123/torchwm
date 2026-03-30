"""Camera sensor that captures RGB frames from a PyBullet physics world.

The camera uses RNGStreams for deterministic jitter/noise on the camera
pose and for any augmentation that needs randomness.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import pybullet as p
except Exception:  # pragma: no cover - tests may not have pybullet
    p = None

from ..api import Sensor, RNGStreams


class CameraSensor(Sensor):
    """Simple RGB camera sensor for PyBullet.

    Config keys:
        - width, height: image dimensions
        - fov: vertical field of view in degrees
        - near, far: clipping planes
        - position: [x,y,z] camera eye position OR
        - target: [x,y,z] point to look at (if not provided, default target is [0,0,0])
        - up: optional up vector
        - jitter: dict with keys 'pos' and/or 'target' specifying positional
                  jitter std-dev in meters applied via RNGStreams.numpy_rng
        - noise: dict with key 'rgb_std' for additive gaussian noise on pixels
    """

    def __init__(
        self,
        config: Dict[str, Any],
        physics_adapter: Any,
        rng: Optional[RNGStreams] = None,
    ):
        if p is None:
            raise ImportError("pybullet is required for CameraSensor")

        self.config = dict(config)
        self._physics = physics_adapter
        self._rng = rng

        self.width = int(config.get("width", 64))
        self.height = int(config.get("height", 64))
        self.fov = float(config.get("fov", 60.0))
        self.near = float(config.get("near", 0.01))
        self.far = float(config.get("far", 100.0))

        self.position = config.get("position", [1.0, 1.0, 1.0])
        self.target = config.get("target", [0.0, 0.0, 0.0])
        self.up = config.get("up", [0.0, 0.0, 1.0])

        self.jitter = config.get("jitter", {})
        self.noise = config.get("noise", {})

    def _apply_jitter(
        self, base: Tuple[float, float, float], key: str
    ) -> Tuple[float, float, float]:
        arr = np.asarray(base, dtype=float)
        if (
            not self.jitter
            or self._rng is None
            or getattr(self._rng, "numpy_rng", None) is None
        ):
            return tuple(arr.tolist())

        std = float(self.jitter.get(key, 0.0))
        if std <= 0.0:
            return tuple(arr.tolist())

        offs = self._rng.numpy_rng.normal(loc=0.0, scale=std, size=3)
        return tuple((arr + offs).tolist())

    def read(self) -> np.ndarray:
        """Capture an RGB frame as a HxWx3 uint8 NumPy array.

        Uses the physics adapter's client id so rendering happens in the same
        physics context. Applies deterministic jitter and optional pixel
        noise controlled by RNGStreams.
        """
        if self._physics is None:
            raise RuntimeError("CameraSensor requires a physics_adapter")

        cid = getattr(self._physics, "_cid", None)
        if cid is None:
            # If the adapter hasn't exposed a client id, fall back to default
            cid = 0

        eye = self._apply_jitter(self.position, "pos")
        target = self._apply_jitter(self.target, "target")
        up = tuple(self.up)

        view_mat = p.computeViewMatrix(
            cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up
        )
        aspect = float(self.width) / float(self.height)
        proj_mat = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=aspect, nearVal=self.near, farVal=self.far
        )

        # Use tiny renderer for headless rendering; request RGB only
        try:
            img = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=view_mat,
                projectionMatrix=proj_mat,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=cid,
            )
        except TypeError:
            # Older pybullet signatures without physicsClientId
            img = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=view_mat,
                projectionMatrix=proj_mat,
                renderer=p.ER_TINY_RENDERER,
            )

        # img[2] is an array of shape (height, width, 4) BGRA in many pybullet builds
        rgba = np.reshape(np.asarray(img[2]), (self.height, self.width, 4))
        # Extract RGB and convert BGR->RGB if necessary. PyBullet often returns RGBA.
        rgb = rgba[:, :, :3].astype(np.uint8)

        # If renderer returns BGR, user can supply conversion later. For now
        # keep the channel order as-is but allow additive pixel noise.
        if (
            self.noise
            and self._rng is not None
            and getattr(self._rng, "numpy_rng", None) is not None
        ):
            rgb_std = float(self.noise.get("rgb_std", 0.0))
            if rgb_std > 0.0:
                noise = self._rng.numpy_rng.normal(0.0, rgb_std, size=rgb.shape)
                rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return rgb
