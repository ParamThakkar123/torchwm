from __future__ import annotations

from .gym_env import GymImageEnv


def make_procgen_env(
    env_name, distribution_mode="hard", num_levels=0, start_level=0, **kwargs
):
    """Factory helper for ProcGen environments."""
    env_id = f"procgen:procgen-{env_name}-v0"
    return GymImageEnv(
        env=env_id,
        distribution_mode=distribution_mode,
        num_levels=num_levels,
        start_level=start_level,
        **kwargs,
    )


def list_available_procgen_envs():
    """List all available ProcGen environment names."""
    return [
        "bigfish",
        "bossfight",
        "caveflyer",
        "chaser",
        "climber",
        "coinrun",
        "dodgeball",
        "fruitbot",
        "heist",
        "jumper",
        "leaper",
        "maze",
        "miner",
        "ninja",
        "plunder",
        "starpilot",
    ]
