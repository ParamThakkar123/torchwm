from __future__ import annotations

import gym
import numpy as np
from PIL import Image


def make_unity_mlagents_env(**kwargs):
    """Factory helper for Unity ML-Agents environments."""
    return UnityMLAgentsEnv(**kwargs)


class UnityMLAgentsEnv:
    """
    Gym-like wrapper for a Unity ML-Agents environment.

    Notes:
    - Supports single-agent control.
    - Supports continuous action spaces.
    - Returns channel-first uint8 images in obs["image"] for Dreamer-style pipelines.
    """

    def __init__(
        self,
        file_name,
        behavior_name=None,
        seed=0,
        size=(64, 64),
        worker_id=0,
        base_port=5005,
        no_graphics=True,
        time_scale=20.0,
        quality_level=1,
        max_episode_steps=1000,
    ):
        from mlagents_envs.base_env import ActionTuple
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import (
            EngineConfigurationChannel,
        )

        self._ActionTuple = ActionTuple
        self._size = (int(size[0]), int(size[1]))
        self._max_episode_steps = int(max_episode_steps)
        self._agent_id = None
        self._last_image = None

        self._engine_channel = EngineConfigurationChannel()
        self._engine_channel.set_configuration_parameters(
            width=self._size[1],
            height=self._size[0],
            quality_level=quality_level,
            time_scale=float(time_scale),
        )

        self._env = UnityEnvironment(
            file_name=file_name,
            seed=seed,
            worker_id=worker_id,
            base_port=base_port,
            no_graphics=no_graphics,
            side_channels=[self._engine_channel],
        )
        self._env.reset()

        behavior_names = list(self._env.behavior_specs.keys())
        if not behavior_names:
            raise ValueError("No Unity behaviors found in the environment.")
        if behavior_name is None:
            behavior_name = behavior_names[0]
        if behavior_name not in self._env.behavior_specs:
            raise ValueError(
                f"Behavior '{behavior_name}' not found. Available: {behavior_names}"
            )

        self._behavior_name = behavior_name
        self._spec = self._env.behavior_specs[self._behavior_name]

        action_spec = self._spec.action_spec
        if not action_spec.is_continuous():
            raise ValueError(
                "UnityMLAgentsEnv currently supports only continuous action spaces."
            )
        self._action_size = int(action_spec.continuous_size)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self._size[0], self._size[1]),
                    dtype=np.uint8,
                )
            }
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._action_size,), dtype=np.float32
        )

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def _extract_agent_data(self, steps, preferred_agent_id):
        agent_ids = np.asarray(getattr(steps, "agent_id", []))
        if agent_ids.size == 0:
            return None

        if preferred_agent_id is None:
            idx = 0
            agent_id = int(agent_ids[idx])
        else:
            matches = np.where(agent_ids == preferred_agent_id)[0]
            if matches.size == 0:
                return None
            idx = int(matches[0])
            agent_id = int(preferred_agent_id)

        obs_list = [np.asarray(obs[idx]) for obs in steps.obs]
        rewards = np.asarray(getattr(steps, "reward", np.zeros(agent_ids.size)))
        reward = float(rewards[idx]) if rewards.size > idx else 0.0

        interrupted = False
        if hasattr(steps, "interrupted"):
            interrupted_arr = np.asarray(steps.interrupted)
            if interrupted_arr.size > idx:
                interrupted = bool(interrupted_arr[idx])

        return agent_id, obs_list, reward, interrupted

    def _vector_to_image(self, vector):
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.zeros_like(arr)

        image = np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        bands = min(arr.size, 8)
        band_w = max(1, self._size[1] // max(1, bands))
        for i in range(bands):
            start = i * band_w
            end = min(self._size[1], start + band_w)
            image[:, start:end, :] = int(255.0 * float(arr[i]))
        return image

    def _to_hwc_uint8(self, obs):
        arr = np.asarray(obs)

        if arr.ndim == 1:
            image = self._vector_to_image(arr)
        elif arr.ndim == 2:
            image = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.ndim == 3:
            image = arr
            # Handle CHW -> HWC if needed.
            if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
                image = image.transpose(1, 2, 0)
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
        else:
            image = np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)

        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size > 0 and image.max() <= 1.0:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)

        if image.shape[0] != self._size[0] or image.shape[1] != self._size[1]:
            image = np.array(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), Image.BILINEAR
                )
            )
        return image

    def _obs_list_to_chw_image(self, obs_list):
        visual = None
        for obs in obs_list:
            arr = np.asarray(obs)
            if arr.ndim == 3:
                visual = arr
                break
        if visual is None and obs_list:
            visual = np.asarray(obs_list[0])
        if visual is None:
            visual = np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        image = self._to_hwc_uint8(visual)
        return image.transpose(2, 0, 1).copy()

    def reset(self):
        self._env.reset()
        decision_steps, terminal_steps = self._env.get_steps(self._behavior_name)

        data = self._extract_agent_data(decision_steps, preferred_agent_id=None)
        if data is None:
            data = self._extract_agent_data(terminal_steps, preferred_agent_id=None)
        if data is None:
            raise RuntimeError("No Unity agents were available after reset.")

        self._agent_id, obs_list, _, _ = data
        image = self._obs_list_to_chw_image(obs_list)
        self._last_image = image
        return {"image": image}

    def step(self, action):
        if self._agent_id is None:
            raise RuntimeError(
                "Environment has terminated. Call reset() before step()."
            )

        action = np.asarray(action, dtype=np.float32).reshape(1, self._action_size)
        action = np.clip(action, -1.0, 1.0)

        self._env.set_actions(
            self._behavior_name,
            self._ActionTuple(continuous=action),
        )
        self._env.step()

        decision_steps, terminal_steps = self._env.get_steps(self._behavior_name)
        terminal_data = self._extract_agent_data(terminal_steps, self._agent_id)

        interrupted = False
        if terminal_data is not None:
            _, obs_list, reward, interrupted = terminal_data
            done = True
            self._agent_id = None
        else:
            decision_data = self._extract_agent_data(decision_steps, self._agent_id)
            if decision_data is None:
                decision_data = self._extract_agent_data(decision_steps, None)
            if decision_data is None:
                raise RuntimeError("No decision step data found after Unity step.")
            self._agent_id, obs_list, reward, _ = decision_data
            done = False

        image = self._obs_list_to_chw_image(obs_list)
        self._last_image = image

        info = {
            "discount": np.array(0.0 if done else 1.0, dtype=np.float32),
            "action": action[0].copy(),
        }
        if done:
            info["interrupted"] = bool(interrupted)

        return {"image": image}, float(reward), bool(done), info

    def render(self, *args, **kwargs):
        if self._last_image is None:
            raise RuntimeError("No frame available. Call reset() before render().")
        return self._last_image.transpose(1, 2, 0).copy()

    def close(self):
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()
