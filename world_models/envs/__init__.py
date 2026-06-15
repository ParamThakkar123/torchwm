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
from .procgen_env import (
    PROCGEN_ENVS,
    ProcgenImageEnv,
    list_procgen_envs,
    make_procgen_env,
    normalize_procgen_env_name,
)
from .brax_env import BraxImageEnv, make_brax_env
from .dmlab import DMLabEnv, make_dmlab_env, DMLAB_LEVELS
from .bsuite_env import BSuiteImageEnv, make_bsuite_env, list_available_bsuite_ids
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
import logging as _logging

logger = _logging.getLogger(__name__)


def make_env(env_id: str, **kwargs):
    """Compatibility helper: create an environment by delegating to
    package-specific factories when available, falling back to gym.make.

    This preserves older callers that expect `make_env` to exist.
    """
    backend = str(kwargs.pop("backend", "")).lower()
    if backend in {"mujoco", "mjcf", "native_mujoco"}:
        return make_mujoco_env(env_id, **kwargs)
    if backend in {"bsuite", "behavior_suite", "behaviour_suite"}:
        return make_bsuite_env(env_id, **kwargs)
    if backend in {"robotics", "gymnasium_robotics"}:
        return make_robotics_env(env_id, **kwargs)
    if backend in {"world-model", "world_model", "model", "wm"}:
        return make_world_model_env(env_id, **kwargs)
    if backend in {"dmlab", "deepmind_lab", "deepmindlab"}:
        return make_dmlab_env(env_id, **kwargs)
    if backend in {"procgen", "coinrun"}:
        return make_procgen_env(env_id, **kwargs)

    # Prefer a package-local factory if present.
    try:
        return make_gym_env(env_id, **kwargs)
    except Exception:
        logger.debug("make_gym_env could not create %s", env_id, exc_info=True)

    try:
        return make_robotics_env(env_id, **kwargs)
    except Exception:
        logger.debug("make_robotics_env could not create %s", env_id, exc_info=True)

    try:
        return make_atari_env(env_id, **kwargs)
    except Exception:
        logger.debug("make_atari_env could not create %s", env_id, exc_info=True)

    try:
        return make_procgen_env(env_id, **kwargs)
    except Exception:
        logger.debug("make_procgen_env could not create %s", env_id, exc_info=True)

    try:
        return make_unity_mlagents_env(env_id, **kwargs)
    except Exception:
        logger.debug(
            "make_unity_mlagents_env could not create %s", env_id, exc_info=True
        )

    try:
        return make_bsuite_env(env_id, **kwargs)
    except Exception:
        logger.debug("make_bsuite_env could not create %s", env_id, exc_info=True)

    # Fall back to gymnasium (formerly gym).
    import gymnasium

    return gymnasium.make(env_id, **kwargs)


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
    "PROCGEN_ENVS",
    "ProcgenImageEnv",
    "list_procgen_envs",
    "make_procgen_env",
    "normalize_procgen_env_name",
    "UnityMLAgentsEnv",
    "make_unity_mlagents_env",
    "DeepMindControlEnv",
    "BraxImageEnv",
    "make_brax_env",
    "DMLabEnv",
    "make_dmlab_env",
    "DMLAB_LEVELS",
    "BSuiteImageEnv",
    "make_bsuite_env",
    "list_available_bsuite_ids",
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
