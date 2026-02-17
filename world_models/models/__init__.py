__all__ = ["Dreamer", "Planet", "DreamerAgent", "JEPAAgent"]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
