"""Observation wrappers for world-model training.

Provides frame stacking, normalization, and other common transformations
for RL and world-model training pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
except Exception:
    torch = None


class ObservationWrapper:
    """Base class for observation transformations."""

    def __init__(self, env: Any):
        self.env = env

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any):
        return self.env.step(action)

    def close(self):
        return self.env.close()


class FrameStackWrapper(ObservationWrapper):
    """Stack consecutive frames for world-model training.

    Args:
        env: underlying env.
        num_stack: number of frames to stack.
        axis: axis to stack on (default 0 for first).
    """

    def __init__(self, env: Any, num_stack: int = 4, axis: int = 0):
        super().__init__(env)
        self.num_stack = num_stack
        self.axis = axis
        self._frames: List[Any] = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._frames = [obs] * self.num_stack
        stacked = self._stack()
        return stacked, info

    def step(self, action: Any):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        if len(self._frames) > self.num_stack:
            self._frames.pop(0)
        stacked = self._stack()
        return stacked, reward, done, info

    def _stack(self) -> Any:
        if torch is not None and isinstance(self._frames[0], np.ndarray):
            try:
                tensors = [torch.from_numpy(f) for f in self._frames]
                return torch.stack(tensors, dim=self.axis)
            except Exception:
                pass
        return np.stack(self._frames, axis=self.axis)


class NormalizeWrapper(ObservationWrapper):
    """Normalize observations to [0, 1] or [-1, 1].

    Args:
        env: underlying env.
        obs_min: min value for normalization (if None, compute from first obs).
        obs_max: max value for normalization (if None, compute from first obs).
        range_min: output min (default 0.0).
        range_max: output max (default 1.0).
    """

    def __init__(
        self,
        env: Any,
        obs_min: Optional[float] = None,
        obs_max: Optional[float] = None,
        range_min: float = 0.0,
        range_max: float = 1.0,
    ):
        super().__init__(env)
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.range_min = range_min
        self.range_max = range_max
        self._computed = False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if not self._computed and self.obs_min is None:
            self.obs_min = float(obs.min()) if hasattr(obs, "min") else 0.0
            self.obs_max = float(obs.max()) if hasattr(obs, "max") else 255.0
            self._computed = True
        normalized = self._normalize(obs)
        return normalized, info

    def step(self, action: Any):
        obs, reward, done, info = self.env.step(action)
        normalized = self._normalize(obs)
        return normalized, reward, done, info

    def _normalize(self, obs: Any) -> Any:
        if self.obs_min is None or self.obs_max is None:
            return obs
        if isinstance(obs, np.ndarray):
            norm = (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-8)
            return norm * (self.range_max - self.range_min) + self.range_min
        return obs


class ResizeWrapper(ObservationWrapper):
    """Resize observations to target dimensions.

    Args:
        env: underlying env.
        target_size: (height, width) tuple.
    """

    def __init__(self, env: Any, target_size: Tuple[int, int]):
        super().__init__(env)
        self.target_size = target_size
        try:
            from PIL import Image

            self._Image = Image
        except Exception:
            self._Image = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        resized = self._resize(obs)
        return resized, info

    def step(self, action: Any):
        obs, reward, done, info = self.env.step(action)
        resized = self._resize(obs)
        return resized, reward, done, info

    def _resize(self, obs: Any) -> Any:
        if self._Image is None or not isinstance(obs, np.ndarray):
            return obs
        h, w = self.target_size
        if obs.ndim == 3 and obs.shape[2] in [1, 3, 4]:
            img = self._Image.fromarray(obs)
            img = img.resize((w, h), self._Image.BILINEAR)
            return np.array(img)
        return obs


class ToTensorWrapper(ObservationWrapper):
    """Convert observations to PyTorch tensors."""

    def __init__(self, env: Any, device: Optional[str] = None):
        super().__init__(env)
        self.device = device or "cpu"

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        tensor = self._to_tensor(obs)
        return tensor, info

    def step(self, action: Any):
        obs, reward, done, info = self.env.step(action)
        tensor = self._to_tensor(obs)
        return tensor, reward, done, info

    def _to_tensor(self, obs: Any) -> Any:
        if torch is None:
            return obs
        if isinstance(obs, np.ndarray):
            t = torch.from_numpy(obs)
            if t.dtype == torch.uint8:
                t = t.float() / 255.0
            return t.to(self.device)
        return obs
