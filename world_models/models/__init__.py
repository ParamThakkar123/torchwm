__all__ = [
    "Dreamer",
    "Planet",
    "DreamerAgent",
    "JEPAAgent",
    "VisionTransformer",
    "ModularRSSM",
    "create_modular_rssm",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
