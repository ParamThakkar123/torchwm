from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

BSUITE_EXAMPLE_IDS = [
    "bandit/0",
    "cartpole/0",
    "catch/0",
    "deep_sea/0",
    "discounting_chain/0",
    "memory_len/0",
    "mnist/0",
    "mountain_car/0",
    "umbrella_chain/0",
]


def list_available_bsuite_ids() -> list[str]:
    """Return the installed BSuite sweep ids, or examples if BSuite is absent."""
    if importlib.util.find_spec("bsuite") is None:
        return list(BSUITE_EXAMPLE_IDS)

    bsuite = importlib.import_module("bsuite")
    sweep = getattr(bsuite, "sweep", None)
    ids = getattr(sweep, "SWEEP", None)
    if ids is None:
        return list(BSUITE_EXAMPLE_IDS)
    return list(ids)


def make_bsuite_env(bsuite_id: str, **kwargs: Any) -> "BSuiteImageEnv":
    """Create a Dreamer-compatible image wrapper around a BSuite task."""
    return BSuiteImageEnv(bsuite_id=bsuite_id, **kwargs)


class _BSuiteDiscreteActionSpace(gym.spaces.Box):
    """Box action space that samples normalized one-hot actions for BSuite."""

    def __init__(self, n: int):
        self.n = int(n)
        super().__init__(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def sample(self, mask: Any | None = None, probability: Any | None = None):
        del mask, probability
        action: np.ndarray = np.full((self.n,), -1.0, dtype=np.float32)
        action[np.random.randint(self.n)] = 1.0
        return action


class BSuiteImageEnv:
    """Gym-like wrapper for DeepMind BSuite ``dm_env`` environments.

    BSuite tasks expose compact ``dm_env`` observations and mostly discrete
    actions. This adapter presents a Gym/Gymnasium-style API with image
    observations under ``{"image": (C, H, W)}`` so TorchWM's pixel-based world
    models can train and evaluate on BSuite diagnostic tasks without requiring
    the base environment to implement rendering.
    """

    def __init__(
        self,
        bsuite_id: str,
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        env: Any | None = None,
    ):
        self.bsuite_id = bsuite_id
        self._seed = int(seed)
        self._size = (int(size[0]), int(size[1]))
        self._env = env if env is not None else self._load_bsuite_env(bsuite_id)
        self._discrete_n: int | None = None
        self._last_time_step: Any | None = None
        self._action_space = self._make_action_space(self._env.action_spec())
        self._observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self._size[0], self._size[1]),
                    dtype=np.uint8,
                )
            }
        )

    @staticmethod
    def _load_bsuite_env(bsuite_id: str):
        if importlib.util.find_spec("bsuite") is None:
            raise ImportError(
                "BSuite support requires the optional 'bsuite' package. "
                "Install it with `pip install bsuite` or `pip install torchwm[bsuite]`."
            )
        bsuite = importlib.import_module("bsuite")
        return bsuite.load_from_id(bsuite_id)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def max_episode_steps(self) -> int:
        return int(getattr(self._env, "bsuite_num_episodes", 0) or 1000)

    def reset(self, seed: int | None = None):
        # BSuite seeds are encoded in the bsuite_id/settings. Accepting seed keeps
        # this wrapper compatible with Gym-style callers.
        if seed is not None:
            self._seed = int(seed)
        time_step = self._env.reset()
        self._last_time_step = time_step
        return self._time_step_to_obs(time_step)

    def step(self, action):
        native_action = self._to_native_action(action)
        time_step = self._env.step(native_action)
        self._last_time_step = time_step
        obs = self._time_step_to_obs(time_step)
        reward = float(time_step.reward or 0.0)
        done = bool(time_step.last())
        info = {
            "discount": np.asarray(
                0.0 if time_step.discount is None else time_step.discount,
                dtype=np.float32,
            ),
            "bsuite_id": self.bsuite_id,
            "action": self._one_hot_action(native_action),
        }
        return obs, reward, done, info

    def render(self, *args: Any, **kwargs: Any):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        if self._last_time_step is None:
            return np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        return self._obs_to_hwc_image(self._last_time_step.observation)

    def close(self):
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def _make_action_space(self, action_spec: Any):
        num_values = getattr(action_spec, "num_values", None)
        if num_values is not None:
            n = int(num_values)
            self._discrete_n = n
            return _BSuiteDiscreteActionSpace(n)

        minimum = np.asarray(getattr(action_spec, "minimum", -1.0), dtype=np.float32)
        maximum = np.asarray(getattr(action_spec, "maximum", 1.0), dtype=np.float32)
        shape = tuple(getattr(action_spec, "shape", minimum.shape or maximum.shape))
        if shape == ():
            shape = (1,)
        self._discrete_n = None
        return gym.spaces.Box(low=minimum, high=maximum, shape=shape, dtype=np.float32)

    def _to_native_action(self, action):
        if self._discrete_n is None:
            return np.asarray(action, dtype=np.float32)
        vec = np.asarray(action, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return 0
        return int(np.argmax(vec[: self._discrete_n]))

    def _one_hot_action(self, action: int):
        if self._discrete_n is None:
            return np.asarray(action, dtype=np.float32)
        out: np.ndarray = np.full((self._discrete_n,), -1.0, dtype=np.float32)
        out[int(action)] = 1.0
        return out

    def _time_step_to_obs(self, time_step: Any) -> dict[str, np.ndarray]:
        return {
            "image": self._obs_to_hwc_image(time_step.observation)
            .transpose(2, 0, 1)
            .copy()
        }

    def _obs_to_hwc_image(self, obs: Any) -> np.ndarray:
        arr = self._flatten_observation(obs)
        if arr.size == 0:
            arr = np.zeros((1,), dtype=np.float32)
        arr = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr)
        if finite.any():
            lo = float(arr[finite].min())
            hi = float(arr[finite].max())
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            else:
                arr = np.zeros_like(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)

        side = int(np.ceil(np.sqrt(arr.size)))
        canvas: np.ndarray = np.zeros((side * side,), dtype=np.float32)
        canvas[: arr.size] = arr
        image = (canvas.reshape(side, side) * 255.0).astype(np.uint8)
        image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[:2] != self._size:
            resampling = getattr(Image, "Resampling", Image)
            image = np.asarray(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), resampling.BILINEAR
                )
            )
        return image

    def _flatten_observation(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            parts = [self._flatten_observation(obs[key]) for key in sorted(obs)]
            return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
        return np.asarray(obs, dtype=np.float32).reshape(-1)
