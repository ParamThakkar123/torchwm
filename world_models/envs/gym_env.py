from __future__ import annotations

import gym
import numpy as np
from PIL import Image


def make_gym_env(env, **kwargs):
    """Factory helper for generic Gym/Gymnasium environments."""
    return GymImageEnv(env=env, **kwargs)


class GymImageEnv:
    """
    Gym-like environment wrapper that always returns image observations.

    - Supports env IDs (string) and prebuilt env objects.
    - For vector observations, it synthesizes an RGB image so pixel-based
      world models can still train.
    - For discrete actions, it exposes a vector action space and maps by argmax.
    """

    def __init__(self, env, seed=0, size=(64, 64), render_mode="rgb_array"):
        self._size = (int(size[0]), int(size[1]))
        self._seed = seed
        self._render_mode = render_mode
        self._seed_applied = False

        if isinstance(env, str):
            self._env = self._make_env_from_id(env, render_mode)
        else:
            self._env = env

        self._last_obs = None
        self._last_image = None

        action_space = getattr(self._env, "action_space", None)
        if action_space is None:
            raise ValueError("Wrapped environment must define action_space.")

        self._discrete_n = int(action_space.n) if hasattr(action_space, "n") else None

        if self._discrete_n is None:
            low = np.asarray(action_space.low, dtype=np.float32)
            high = np.asarray(action_space.high, dtype=np.float32)
            self._action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self._action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self._discrete_n,), dtype=np.float32
            )
            self._action_space.sample = self._sample_discrete_action

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

    def _make_env_from_id(self, env_id, render_mode):
        # Prefer gymnasium if available, then fallback to gym.
        try:
            import gymnasium as gymnasium

            try:
                return gymnasium.make(env_id, render_mode=render_mode)
            except TypeError:
                return gymnasium.make(env_id)
        except Exception:
            try:
                return gym.make(env_id, render_mode=render_mode)
            except TypeError:
                return gym.make(env_id)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def max_episode_steps(self):
        if (
            hasattr(self._env, "_max_episode_steps")
            and self._env._max_episode_steps is not None
        ):
            return int(self._env._max_episode_steps)
        if (
            getattr(self._env, "spec", None) is not None
            and getattr(self._env.spec, "max_episode_steps", None) is not None
        ):
            return int(self._env.spec.max_episode_steps)
        return 1000

    def _sample_discrete_action(self):
        idx = np.random.randint(0, self._discrete_n)
        action = -np.ones((self._discrete_n,), dtype=np.float32)
        action[idx] = 1.0
        return action

    def _vector_to_image(self, vector):
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        vmin = float(vec.min())
        vmax = float(vec.max())
        if vmax > vmin:
            vec = (vec - vmin) / (vmax - vmin)
        else:
            vec = np.zeros_like(vec)

        image = np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        bands = min(8, vec.size)
        band_w = max(1, self._size[1] // max(1, bands))
        for i in range(bands):
            start = i * band_w
            end = min(self._size[1], start + band_w)
            image[:, start:end, :] = int(255.0 * float(vec[i]))
        return image

    def _obs_to_hwc_image(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]

        if isinstance(obs, dict):
            for key in ("image", "pixels", "rgb", "observation"):
                if key in obs:
                    candidate = np.asarray(obs[key])
                    if candidate.ndim in (1, 2, 3):
                        return self._obs_to_hwc_image(candidate)
            for value in obs.values():
                candidate = np.asarray(value)
                if candidate.ndim in (1, 2, 3):
                    return self._obs_to_hwc_image(candidate)
            return None

        arr = np.asarray(obs)
        if arr.ndim == 1:
            image = self._vector_to_image(arr)
        elif arr.ndim == 2:
            image = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.ndim == 3:
            image = arr
            if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
                image = image.transpose(1, 2, 0)
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
        else:
            return None

        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size > 0 and image.max() <= 1.0:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)
        return image

    def _render_hwc_image(self, last_obs=None):
        frame = None
        try:
            frame = self._env.render()
        except Exception:
            frame = None
        if isinstance(frame, tuple):
            frame = frame[0]
        if isinstance(frame, np.ndarray):
            return frame

        try:
            frame = self._env.render(mode=self._render_mode)
        except Exception:
            frame = None
        if isinstance(frame, tuple):
            frame = frame[0]
        if isinstance(frame, np.ndarray):
            return frame

        if last_obs is not None:
            return self._obs_to_hwc_image(last_obs)
        return None

    def _to_chw_uint8_image(self, obs):
        image = self._obs_to_hwc_image(obs)
        if image is None:
            image = self._render_hwc_image(last_obs=obs)
        if image is None:
            raise RuntimeError(
                "Failed to obtain an RGB frame from environment observation or render()."
            )

        if image.shape[0] != self._size[0] or image.shape[1] != self._size[1]:
            image = np.array(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), Image.BILINEAR
                )
            )
        return image.transpose(2, 0, 1).copy()

    def _to_native_action(self, action):
        if self._discrete_n is None:
            action = np.asarray(action, dtype=np.float32)
            low = np.asarray(self._env.action_space.low, dtype=np.float32)
            high = np.asarray(self._env.action_space.high, dtype=np.float32)
            clipped = np.clip(action, low, high).astype(np.float32)
            return clipped, clipped

        vec = np.asarray(action, dtype=np.float32).reshape(-1)
        if vec.size == self._discrete_n and vec.size > 1:
            idx = int(np.argmax(vec))
        elif vec.size >= 1:
            idx = int(round(float(vec[0])))
        else:
            idx = 0
        idx = int(np.clip(idx, 0, self._discrete_n - 1))
        encoded = -np.ones((self._discrete_n,), dtype=np.float32)
        encoded[idx] = 1.0
        return idx, encoded

    def reset(self):
        if not self._seed_applied:
            try:
                result = self._env.reset(seed=self._seed)
            except TypeError:
                result = self._env.reset()
            self._seed_applied = True
        else:
            result = self._env.reset()

        obs = result[0] if isinstance(result, tuple) else result
        self._last_obs = obs
        image = self._to_chw_uint8_image(obs)
        self._last_image = image
        return {"image": image}

    def step(self, action):
        native_action, model_action = self._to_native_action(action)
        result = self._env.step(native_action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
            done = bool(done)

        if info is None:
            info = {}
        info = dict(info)
        if "discount" not in info:
            info["discount"] = np.array(0.0 if done else 1.0, dtype=np.float32)
        info["action"] = np.asarray(model_action, dtype=np.float32).copy()

        self._last_obs = obs
        image = self._to_chw_uint8_image(obs)
        self._last_image = image
        return {"image": image}, float(reward), done, info

    def render(self, *args, **kwargs):
        frame = self._render_hwc_image(last_obs=self._last_obs)
        if frame is None:
            if self._last_image is None:
                raise RuntimeError("No frame available. Call reset() before render().")
            return self._last_image.transpose(1, 2, 0).copy()
        return frame

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()
