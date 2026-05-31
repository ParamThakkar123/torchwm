"""Small, import-safe catalog of available environments and backends.

This module replaces the previous `world_models.ui.catalog` and is safe to
import from lightweight CLI tools and tests without pulling in any UI
dependencies.
"""

from __future__ import annotations

from typing import Any

from world_models.envs import list_available_atari_envs


DREAMER_ENVS = [
    "cartpole-balance",
    "cartpole-swingup",
    "cheetah-run",
    "finger-spin",
    "reacher-easy",
    "walker-walk",
    "walker-run",
    "quadruped-walk",
]

PLANET_BASE_ENVS = [
    "CartPole-v1",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "Acrobot-v1",
    "HalfCheetah-v4",
    "Humanoid-v4",
]

GYM_ENVS = [
    "CartPole-v1",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "Acrobot-v1",
    "HalfCheetah-v4",
    "Humanoid-v4",
    "Hopper-v4",
    "Swimmer-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Reacher-v4",
    "Pusher-v4",
    "Manipulator-v4",
    "LunarLander-v3",
    "LunarLanderContinuous-v3",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
    "CarRacing-v3",
    "Blackjack-v1",
    "FrozenLake-v1",
    "FrozenLake8x8-v1",
    "Taxi-v3",
    "InvertedPendulum-v4",
    "InvertedDoublePendulum-v4",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Swimmer-v2",
    "Walker2d-v2",
    "Reacher-v2",
    "Pusher-v2",
]

UNITY_ENVS: list[str] = []

ATARI_ENVS: list[str] = []
try:
    ATARI_ENVS = list_available_atari_envs()
except Exception:
    ATARI_ENVS = []


ENV_BACKENDS: dict[str, dict[str, Any]] = {
    "dm_control": {
        "label": "DM Control",
        "description": "DeepMind Control Suite",
        "environments": DREAMER_ENVS,
    },
    "mujoco": {
        "label": "MuJoCo",
        "description": "MuJoCo physics environments",
        "environments": GYM_ENVS,
    },
    "gym": {
        "label": "Gym",
        "description": "OpenAI Gym environments",
        "environments": PLANET_BASE_ENVS,
    },
    "unity": {
        "label": "Unity ML Agents",
        "description": "Unity ML Agents environments",
        "environments": UNITY_ENVS,
    },
    "atari": {
        "label": "Atari",
        "description": "Atari 2600 environments via ALE",
        "environments": ATARI_ENVS,
    },
}


def _build_env_catalog() -> dict[str, list[str]]:
    atari_envs: list[str] = []
    try:
        atari_envs = list_available_atari_envs()
    except Exception:
        atari_envs = []
    return {
        "dreamerv1": DREAMER_ENVS + GYM_ENVS,
        "dreamerv2": DREAMER_ENVS + GYM_ENVS,
        "planet": PLANET_BASE_ENVS + atari_envs[:80],
        "iris": atari_envs[:80],
    }


ENVIRONMENTS_BY_MODEL = _build_env_catalog()
