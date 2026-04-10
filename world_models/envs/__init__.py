import importlib.util

from .ale_atari_env import make_atari_env, list_available_atari_envs
from .ale_atari_vector_env import make_atari_vector_env
from .mujoco_env import make_humanoid_env, make_half_cheetah_env
from .gym_env import GymImageEnv, make_gym_env
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
from .vector_env import (
    SimWorker,
    VectorizedEnv,
    TorchVectorizedEnv,
    GPUVectorizedEnv,
    IsaacLabObsWrapper,
)

_isaaclab_available = importlib.util.find_spec("isaaclab") is not None
_cuda_kernels_available = (
    importlib.util.find_spec("torch.utils.cpp_extension") is not None
)
from .vector_env import (
    SimWorker,
    VectorizedEnv,
    TorchVectorizedEnv,
    GPUVectorizedEnv,
    IsaacLabObsWrapper,
)

_isaaclab_available = importlib.util.find_spec("isaaclab") is not None
if _cuda_kernels_available:
    try:
        from .cuda_kernels import batched_normalize, batched_add_noise

        _has_cuda_kernels = True
    except ImportError:
        _has_cuda_kernels = False
else:
    _has_cuda_kernels = False

__all__ = [
    "make_atari_env",
    "list_available_atari_envs",
    "make_atari_vector_env",
    "make_humanoid_env",
    "make_half_cheetah_env",
    "GymImageEnv",
    "make_gym_env",
    "UnityMLAgentsEnv",
    "make_unity_mlagents_env",
    "DeepMindControlEnv",
    "SimWorker",
    "VectorizedEnv",
    "TorchVectorizedEnv",
    "GPUVectorizedEnv",
    "IsaacLabObsWrapper",
    "TimeLimit",
    "ActionRepeat",
    "NormalizeActions",
    "ObsDict",
    "OneHotAction",
    "RewardObs",
    "ResizeImage",
    "RenderImage",
    "SelectAction",
]

if _isaaclab_available:
    from .isaaclab_env import IsaacLabImageEnv, make_isaaclab_env

    __all__.extend(["IsaacLabImageEnv", "make_isaaclab_env"])

if _has_cuda_kernels:
    __all__.extend(["batched_normalize", "batched_add_noise"])
