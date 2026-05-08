"""
TorchWM - Modular PyTorch Library for World Models

Public API exports for easy imports.

Usage:
    from world_models import DreamerAgent, DreamerConfig
    agent = DreamerAgent(DreamerConfig())
    agent.train()
"""

__version__ = "0.3.2"


def __getattr__(name):
    """Lazy import to avoid loading unused modules and their dependencies."""
    # Models & Agents
    if name in (
        "Dreamer",
        "Planet",
        "DreamerAgent",
        "JEPAAgent",
        "VisionTransformer",
        "ModularRSSM",
        "create_modular_rssm",
    ):
        from world_models import models as _models

        return getattr(_models, name)

    # Configs
    if name in (
        "DreamerConfig",
        "JEPAConfig",
        "DiTConfig",
        "get_dit_config",
        "DiamondConfig",
        "IRISConfig",
        "ATARI_100K_GAMES",
        "HUMAN_SCORES",
        "RANDOM_SCORES",
    ):
        from world_models import configs as _configs

        return getattr(_configs, name)

    # Environments
    if name in (
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
        "TimeLimit",
        "ActionRepeat",
        "NormalizeActions",
        "ObsDict",
        "OneHotAction",
        "RewardObs",
        "ResizeImage",
        "RenderImage",
        "SelectAction",
    ):
        from world_models import envs as _envs

        return getattr(_envs, name)

    # Inference Operators
    if name in (
        "OperatorABC",
        "DreamerOperator",
        "JEPAOperator",
        "IrisOperator",
        "PlaNetOperator",
        "get_operator",
    ):
        from world_models.inference import operators as _ops

        return getattr(_ops, name)

    # Reward Models
    if name in ("RewardModel", "ValueModel"):
        from world_models import reward as _reward

        return getattr(_reward, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Models & Agents
    "Dreamer",
    "Planet",
    "DreamerAgent",
    "JEPAAgent",
    "VisionTransformer",
    "ModularRSSM",
    "create_modular_rssm",
    # Configs
    "DreamerConfig",
    "JEPAConfig",
    "DiTConfig",
    "get_dit_config",
    "DiamondConfig",
    "IRISConfig",
    "ATARI_100K_GAMES",
    "HUMAN_SCORES",
    "RANDOM_SCORES",
    # Environments
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
    "TimeLimit",
    "ActionRepeat",
    "NormalizeActions",
    "ObsDict",
    "OneHotAction",
    "RewardObs",
    "ResizeImage",
    "RenderImage",
    "SelectAction",
    # Inference Operators
    "OperatorABC",
    "DreamerOperator",
    "JEPAOperator",
    "IrisOperator",
    "PlaNetOperator",
    # Reward Models
    "RewardModel",
    "ValueModel",
]
