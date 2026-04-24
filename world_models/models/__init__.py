__all__ = [
    "Dreamer",
    "Planet",
    "DreamerAgent",
    "JEPAAgent",
    "VisionTransformer",
    "ModularRSSM",
    "create_modular_rssm",
    "Genie",
    "LatentActionModel",
    "DynamicsModel",
    "create_genie",
    "create_genie_small",
    "create_genie_large",
    "create_latent_action_model",
    "create_dynamics_model",
]


def __getattr__(name):
    if name == "Dreamer":
        from .dreamer import Dreamer

        return Dreamer
    if name == "Planet":
        from .planet import Planet

        return Planet
    if name == "DreamerAgent":
        from .dreamer import DreamerAgent

        return DreamerAgent
    if name == "JEPAAgent":
        from .jepa_agent import JEPAAgent

        return JEPAAgent
    if name == "VisionTransformer":
        from .vit import VisionTransformer

        return VisionTransformer
    if name == "ModularRSSM":
        from .modular_rssm import ModularRSSM

        return ModularRSSM
    if name == "create_modular_rssm":
        from .modular_rssm import create_modular_rssm

        return create_modular_rssm
    if name == "Genie":
        from .genie import Genie

        return Genie
    if name == "LatentActionModel":
        from .latent_action_model import LatentActionModel

        return LatentActionModel
    if name == "DynamicsModel":
        from .dynamics_model import DynamicsModel

        return DynamicsModel
    if name == "create_genie":
        from .genie import create_genie

        return create_genie
    if name == "create_genie_small":
        from .genie import create_genie_small

        return create_genie_small
    if name == "create_genie_large":
        from .genie import create_genie_large

        return create_genie_large
    if name == "create_latent_action_model":
        from .latent_action_model import create_latent_action_model

        return create_latent_action_model
    if name == "create_dynamics_model":
        from .dynamics_model import create_dynamics_model

        return create_dynamics_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
