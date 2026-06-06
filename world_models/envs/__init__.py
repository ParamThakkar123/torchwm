from .ale_atari_env import make_atari_env, list_available_atari_envs
from .ale_atari_vector_env import make_atari_vector_env
from .robotics_env import (
    list_gymnasium_robotics_envs,
    make_robotics_env,
    register_gymnasium_robotics_envs,
)
from .mujoco_env import (
    MuJoCoImageEnv,
    make_mujoco_env,
    make_mujoco_env_from_config,
)
from .gym_env import GymImageEnv, make_gym_env
from .world_model_env import WorldModelEnv, make_world_model_env
from .brax_env import BraxImageEnv, make_brax_env
from .unity_env import UnityMLAgentsEnv, make_unity_mlagents_env
from .wrappers import (
    TimeLimit,
    ActionRepeat,
    NormalizeActions,
    ObsDict,
    OneHotAction,
    RewardObs,
    ResizeImage,
    RenderImage,
    SelectAction,
)
from .dmc import DeepMindControlEnv
import gym


def make_env(env_id: str, **kwargs):
    """Compatibility helper: create an environment by delegating to
    package-specific factories when available, falling back to gym.make.

    This preserves older callers that expect `make_env` to exist.
    """
    backend = str(kwargs.pop("backend", "")).lower()
    if backend in {"mujoco", "mjcf", "native_mujoco"}:
        return make_mujoco_env(env_id, **kwargs)
    if backend in {"robotics", "gymnasium_robotics"}:
        return make_robotics_env(env_id, **kwargs)
    if backend in {"world-model", "world_model", "model", "wm"}:
        return make_world_model_env(env_id, **kwargs)

    # Prefer a package-local factory if present.
    try:
        return make_gym_env(env_id, **kwargs)
    except Exception:
        pass

    try:
        return make_robotics_env(env_id, **kwargs)
    except Exception:
        pass

    try:
        return make_atari_env(env_id, **kwargs)
    except Exception:
        pass

    try:
        return make_unity_mlagents_env(env_id, **kwargs)
    except Exception:
        pass

    # Fall back to gym.
    return gym.make(env_id, **kwargs)


__all__ = [
    "make_atari_env",
    "list_available_atari_envs",
    "make_atari_vector_env",
    "MuJoCoImageEnv",
    "make_mujoco_env",
    "make_mujoco_env_from_config",
    "list_gymnasium_robotics_envs",
    "make_robotics_env",
    "register_gymnasium_robotics_envs",
    "GymImageEnv",
    "make_gym_env",
    "WorldModelEnv",
    "make_world_model_env",
    "UnityMLAgentsEnv",
    "make_unity_mlagents_env",
    "DeepMindControlEnv",
    "BraxImageEnv",
    "make_brax_env",
    "TimeLimit",
    "ActionRepeat",
    "NormalizeActions",
    "ObsDict",
    "OneHotAction",
    "RewardObs",
    "ResizeImage",
    "RenderImage",
    "SelectAction",
    "make_env",
]
