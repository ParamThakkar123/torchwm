"""Training modules for World Models."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "train_convae":
        from world_models.training.train_convvae import train_convae

        return train_convae
    if name == "train_mdn_rnn":
        from world_models.training.train_mdn_rnn import train_mdn_rnn

        return train_mdn_rnn
    if name == "train_controller":
        from world_models.training.train_controller import train_controller

        return train_controller
    if name == "DiamondAgent":
        from world_models.training.train_diamond import DiamondAgent

        return DiamondAgent
    if name == "train_diamond":
        from world_models.training.train_diamond import train_diamond

        return train_diamond
    if name == "train_dreamer":
        from world_models.training.train_dreamer import train_dreamer

        return train_dreamer
    if name == "GenieTrainer":
        from world_models.training.train_genie import GenieTrainer

        return GenieTrainer
    if name == "create_genie_trainer":
        from world_models.training.train_genie import create_genie_trainer

        return create_genie_trainer
    if name == "IRISTrainer":
        from world_models.training.train_iris import IRISTrainer

        return IRISTrainer
    if name == "train_planet":
        from world_models.training.train_planet import train as train_planet

        return train_planet
    if name == "train_rssm":
        from world_models.training.train_rssm import train_rssm

        return train_rssm
    if name == "run_atari_100k":
        from world_models.training.run_atari_100k import run_atari_100k

        return run_atari_100k
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiamondAgent",
    "GenieTrainer",
    "IRISTrainer",
    "create_genie_trainer",
    "run_atari_100k",
    "train_convae",
    "train_controller",
    "train_diamond",
    "train_dreamer",
    "train_mdn_rnn",
    "train_planet",
    "train_rssm",
]
