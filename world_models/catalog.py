"""Small, import-safe catalog of available environments and backends.

This module replaces the previous `world_models.ui.catalog` and is safe to
import from lightweight CLI tools and tests without pulling in any UI
dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EnvBackendSpec:
    """Public metadata for an environment backend exposed by TorchWM.

    The spec is intentionally lightweight and import-safe so CLI commands, docs,
    and integrations can enumerate backends without importing heavy simulation
    packages. ``env_backend`` is the value used by DreamerConfig when it differs
    from the catalog key.
    """

    key: str
    label: str
    description: str
    environments: tuple[str, ...]
    env_backend: str
    aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return the legacy dictionary representation used by older callers."""
        data = asdict(self)
        data["environments"] = list(self.environments)
        data["aliases"] = list(self.aliases)
        return data


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

DMLAB_ENVS = [
    "rooms_collect_good_objects_train",
    "rooms_collect_good_objects_test",
    "rooms_exploit_deferred_effects_train",
    "rooms_exploit_deferred_effects_test",
    "rooms_select_nonmatching_object",
    "rooms_watermaze",
    "rooms_keys_doors_puzzle",
    "language_select_described_object",
    "language_select_located_object",
    "language_execute_random_task",
    "nav_maze_static_01",
    "nav_maze_static_02",
    "nav_maze_random_goal_01",
    "nav_maze_random_goal_02",
    "lt_chasm",
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


def _list_available_atari_envs() -> list[str]:
    """Return registered Atari ids without making catalog imports heavy."""
    try:
        from world_models.envs import list_available_atari_envs

        return list_available_atari_envs()
    except Exception:
        return []


ATARI_ENVS: list[str] = _list_available_atari_envs()


ENV_BACKEND_SPECS: tuple[EnvBackendSpec, ...] = (
    EnvBackendSpec(
        key="dm_control",
        label="DM Control",
        description="DeepMind Control Suite",
        environments=tuple(DREAMER_ENVS),
        env_backend="dmc",
        aliases=("dmc",),
    ),
    EnvBackendSpec(
        key="dmlab",
        label="DeepMind Lab",
        description="DeepMind Lab 3D navigation and puzzle tasks",
        environments=tuple(DMLAB_ENVS),
        env_backend="dmlab",
        aliases=("deepmind_lab", "deepmindlab"),
    ),
    EnvBackendSpec(
        key="mujoco",
        label="MuJoCo",
        description="MuJoCo physics environments",
        environments=tuple(GYM_ENVS),
        env_backend="mujoco",
        aliases=("mjcf", "native_mujoco"),
    ),
    EnvBackendSpec(
        key="gym",
        label="Gym",
        description="OpenAI Gym environments",
        environments=tuple(PLANET_BASE_ENVS),
        env_backend="gym",
        aliases=("gymnasium", "generic"),
    ),
    EnvBackendSpec(
        key="robotics",
        label="Gymnasium Robotics",
        description="Gymnasium Robotics and legacy MuJoCo v2/v3 environments",
        environments=tuple(ROBOTICS_ENVS),
        env_backend="robotics",
        aliases=("gymnasium_robotics",),
    ),
    EnvBackendSpec(
        key="unity",
        label="Unity ML Agents",
        description="Unity ML Agents environments",
        environments=tuple(UNITY_ENVS),
        env_backend="unity_mlagents",
        aliases=("unity", "mlagents"),
    ),
    EnvBackendSpec(
        key="atari",
        label="Atari",
        description="Atari 2600 environments via ALE",
        environments=tuple(ATARI_ENVS),
        env_backend="gym",
        aliases=("ale",),
    ),
)

ENV_BACKENDS: dict[str, dict[str, Any]] = {
    spec.key: spec.as_dict() for spec in ENV_BACKEND_SPECS
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
    dreamer_envs = _dedupe_envs(DREAMER_ENVS, DMLAB_ENVS, general_control_envs)
    planet_envs = _dedupe_envs(PLANET_BASE_ENVS, atari_envs[:80], robotics_envs)
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


__all__ = [
    "EnvBackendSpec",
    "ENV_BACKEND_SPECS",
    "ENV_BACKENDS",
    "ENVIRONMENTS_BY_MODEL",
    "DMLAB_ENVS",
]
