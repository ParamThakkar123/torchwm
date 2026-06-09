"""Small, import-safe catalog of available environments and backends.

This module replaces the previous `world_models.ui.catalog` and is safe to
import from lightweight CLI tools and tests without pulling in any UI
dependencies.
"""

from __future__ import annotations

from typing import Any

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


def _list_available_robotics_envs() -> list[str]:
    """Return all Gymnasium Robotics ids when the optional package exists."""
    try:
        from world_models.envs import list_gymnasium_robotics_envs

        return list_gymnasium_robotics_envs()
    except Exception:
        return []


ROBOTICS_ENVS: list[str] = _list_available_robotics_envs()

UNITY_ENVS: list[str] = []

PROCGEN_ENVS = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


def _list_available_bsuite_ids() -> list[str]:
    """Return BSuite sweep ids or examples without making catalog imports heavy."""
    try:
        from world_models.envs import list_available_bsuite_ids

        return list_available_bsuite_ids()
    except Exception:
        return [
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


BSUITE_ENVS: list[str] = _list_available_bsuite_ids()


def _list_available_atari_envs() -> list[str]:
    """Return registered Atari ids without making catalog imports heavy."""
    try:
        from world_models.envs import list_available_atari_envs

        return list_available_atari_envs()
    except Exception:
        return []


ATARI_ENVS: list[str] = _list_available_atari_envs()


def _list_available_dmlab_levels() -> list[str]:
    try:
        from world_models.envs.dmlab import DMLAB_LEVELS

        return list(DMLAB_LEVELS)
    except Exception:
        return []


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
    "robotics": {
        "label": "Gymnasium Robotics",
        "description": "Gymnasium Robotics and legacy MuJoCo v2/v3 environments",
        "environments": ROBOTICS_ENVS,
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
    "procgen": {
        "label": "Procgen",
        "description": "Procedurally generated benchmark games",
        "environments": PROCGEN_ENVS,
    },
    "bsuite": {
        "label": "BSuite",
        "description": "DeepMind Behaviour Suite diagnostic RL tasks",
        "environments": BSUITE_ENVS,
    },
    "dmlab": {
        "label": "DeepMind Lab",
        "description": "DeepMind Lab 3D navigation and puzzle tasks",
        "environments": _list_available_dmlab_levels(),
    },
}


def _dedupe_envs(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    combined: list[str] = []
    for group in groups:
        for env_id in group:
            if env_id not in seen:
                combined.append(env_id)
                seen.add(env_id)
    return combined


def _build_env_catalog() -> dict[str, list[str]]:
    atari_envs = _list_available_atari_envs()
    robotics_envs = _list_available_robotics_envs()
    general_control_envs = _dedupe_envs(GYM_ENVS, robotics_envs)
    atari_and_robotics_envs = _dedupe_envs(atari_envs[:80], robotics_envs)
    bsuite_envs = _list_available_bsuite_ids()
    dmlab_envs = _list_available_dmlab_levels()
    dreamer_envs = _dedupe_envs(
        DREAMER_ENVS, general_control_envs, PROCGEN_ENVS, bsuite_envs, dmlab_envs
    )
    planet_envs = _dedupe_envs(
        PLANET_BASE_ENVS, atari_envs[:80], robotics_envs, PROCGEN_ENVS
    )
    return {
        "dreamer": dreamer_envs,
        "dreamerv1": dreamer_envs,
        "dreamerv2": dreamer_envs,
        "planet": planet_envs,
        "rssm": planet_envs,
        "iris": atari_and_robotics_envs,
        "diamond": atari_and_robotics_envs,
        "genie": atari_and_robotics_envs,
        "dit": atari_and_robotics_envs,
        # I-JEPA/JEPA is intentionally omitted because it trains on image datasets
        # rather than online Gymnasium environments.
    }


ENVIRONMENTS_BY_MODEL = _build_env_catalog()
