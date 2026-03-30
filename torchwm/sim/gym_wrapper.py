"""Gymnasium-compatible wrapper around BasicEnv.

Provides observation_space and action_space mapping so trainers can plug
into standard RL libraries. Supports multi-sensor observations via
gym.spaces.Dict and per-joint action ranges.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - gym optional
    import gym  # type: ignore[no-redef]
    from gym import spaces  # type: ignore[no-redef]


class GymWrapperEnv(gym.Env):  # type: ignore[valid-type, misc]
    """Wrap a BasicEnv into a Gymnasium-compatible Env.

    Args:
        env: a BasicEnv instance (must expose `.config` dict and have sensors).
        sensors: list of sensor keys to include in observation. Currently
            supports 'camera' (RGB). Default: ['camera'].
        action_config: dict with keys:
            - 'type': 'torque' or 'position'
            - 'body_index': int (which body to control, default 0 = first)
            - 'joint_indices': list of joint indices (default all joints)
            - 'low': low bound for actions (default -1.0)
            - 'high': high bound for actions (default 1.0)
        If action_config is None, uses a noop Discrete(1) action space.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        env: Any,
        sensors: Optional[List[str]] = None,
        action_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._env = env
        self.sensors = sensors or ["camera"]
        self.action_config = action_config

        # Build observation space
        self.observation_space = self._build_observation_space()

        # Build action space
        self.action_space = self._build_action_space()

    def _build_observation_space(self) -> "spaces.Space":  # type: ignore[return-value]
        """Build observation space from configured sensors."""
        cam_cfg = getattr(self._env, "config", {}).get("camera", {})
        h = int(cam_cfg.get("height", 64))
        w = int(cam_cfg.get("width", 64))
        c = 3

        cam_space = spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)  # type: ignore[assignment]

        if len(self.sensors) == 1 and self.sensors[0] == "camera":
            return cam_space  # type: ignore[return-value]
        else:
            obs_spaces: Dict[str, Any] = {}
            if "camera" in self.sensors:
                obs_spaces["camera"] = cam_space  # type: ignore[index]
            # Add more sensors here as they are implemented
            return spaces.Dict(obs_spaces)  # type: ignore[arg-type]

    def _build_action_space(self) -> "spaces.Space":  # type: ignore[return-value]
        """Build action space from action_config."""
        if self.action_config is None:
            return spaces.Discrete(1)  # type: ignore[return-value]

        act_type = self.action_config.get("type", "torque")
        low = float(self.action_config.get("low", -1.0))
        high = float(self.action_config.get("high", 1.0))

        # Determine number of action dimensions
        n_joints = self._get_num_joints()
        if n_joints is None or n_joints == 0:
            n_joints = 1

        return spaces.Box(low=low, high=high, shape=(n_joints,), dtype=np.float32)  # type: ignore[return-value]

    def _get_num_joints(self) -> Optional[int]:
        """Query physics adapter for number of controllable joints."""
        try:
            body_list = getattr(self._env._physics, "_body_uids", [])
            if not body_list:
                return None
            body_id = body_list[0]
            import pybullet as p

            n_joints = p.getNumJoints(body_id)
            return n_joints
        except Exception:
            return None

    def _get_joint_indices(self) -> List[int]:
        """Get list of joint indices to control."""
        if self.action_config is None:
            return []
        return self.action_config.get(
            "joint_indices", list(range(self._get_num_joints() or 0))
        )

    def _map_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Map a gym action to the env's action dict format."""
        if self.action_config is None:
            return {}

        body_list = getattr(self._env._physics, "_body_uids", [])
        body_idx = self.action_config.get("body_index", 0)
        body = (
            int(body_list[body_idx])
            if body_list and body_idx < len(body_list)
            else None
        )

        joint_indices = self._get_joint_indices()
        # Select subset of action vector for specified joints
        if len(joint_indices) > 0 and len(action) >= len(joint_indices):
            selected = [float(action[i]) for i in range(len(joint_indices))]
        else:
            selected = [float(a) for a in action]

        act_type = self.action_config.get("type", "torque")
        return {"body": body, "action": selected, "mode": act_type}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self._env.reset(seed=seed, options=options)
        return self._wrap_observation(obs), info

    def step(self, action):
        env_action = self._map_action(np.asarray(action, dtype=np.float32))
        obs, reward, done, info = self._env.step(env_action)
        return self._wrap_observation(obs), reward, done, info

    def _wrap_observation(self, obs: Any) -> Any:
        """Wrap raw observation into the configured observation space."""
        if isinstance(self.observation_space, spaces.Dict):
            wrapped = {}
            if "camera" in self.sensors:
                wrapped["camera"] = obs
            return wrapped
        return obs

    def render(self, mode: str = "rgb_array"):
        return self._env.render(mode=mode)

    def close(self):
        return self._env.close()


def make_gym_env(
    config: Dict[str, Any],
    seed: Optional[int] = None,
    sensors: Optional[List[str]] = None,
    action_config: Optional[Dict[str, Any]] = None,
) -> GymWrapperEnv:
    """Factory helper to create a GymWrapperEnv from a config dict.

    Args:
        config: BasicEnv configuration dict.
        seed: optional seed for environment reset.
        sensors: list of sensor keys for observation.
        action_config: action configuration dict.

    Returns:
        GymWrapperEnv instance ready for training.
    """
    from .envs.basic_env import BasicEnv

    env = BasicEnv(config)
    return GymWrapperEnv(env, sensors=sensors, action_config=action_config)
