from .base import OperatorABC
from .dreamer_operator import DreamerOperator
from .jepa_operator import JEPAOperator
from .iris_operator import IrisOperator
from .planet_operator import PlaNetOperator


def get_operator(name: str, **kwargs):
    """Factory function to get inference operators by name.

    Args:
        name: One of 'dreamer', 'jepa', 'iris', 'planet'
        **kwargs: Operator-specific configuration

    Returns:
        Configured OperatorABC instance

    Example:
        >>> op = get_operator('dreamer', image_size=64, action_dim=6)
        >>> processed = op.process({'image': image, 'action': action})
    """
    operators = {
        "dreamer": DreamerOperator,
        "jepa": JEPAOperator,
        "iris": IrisOperator,
        "planet": PlaNetOperator,
    }
    if name.lower() not in operators:
        raise ValueError(
            f"Unknown operator {name!r}. Available: {list(operators.keys())}"
        )
    return operators[name.lower()](**kwargs)


__all__ = [
    "OperatorABC",
    "DreamerOperator",
    "JEPAOperator",
    "IrisOperator",
    "PlaNetOperator",
    "get_operator",
]
