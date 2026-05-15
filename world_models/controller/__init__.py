"""
Controller sub-module - Policies and rollout generators.

Exported Components:
    - RSSMPolicy: Model-predictive controller with RSSM latent model
    - RolloutGenerator: Episode rollout generator for evaluation
    - IRISActor: Actor network for IRIS policy
    - IRISValueNetwork: Value network for IRIS
"""

__all__ = [
    "RSSMPolicy",
    "RolloutGenerator",
    "IRISActor",
    "IRISCritic",
    "IRISPolicy",
    "CNNFeatureExtractor",
]


def __getattr__(name):
    if name == "RSSMPolicy":
        from .rssm_policy import RSSMPolicy

        return RSSMPolicy
    if name == "RolloutGenerator":
        from .rollout_generator import RolloutGenerator

        return RolloutGenerator
    if name == "IRISActor":
        from .iris_policy import IRISActor

        return IRISActor
    if name == "IRISCritic":
        from .iris_policy import IRISCritic

        return IRISCritic
    if name == "IRISPolicy":
        from .iris_policy import IRISPolicy

        return IRISPolicy
    if name == "CNNFeatureExtractor":
        from .iris_policy import CNNFeatureExtractor

        return CNNFeatureExtractor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
