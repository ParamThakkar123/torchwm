from typing import Any, Optional
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def make_atari_env(
    env_id: str,
    obs_type: str = "rgb",
    frameskip: int = 4,
    repeat_action_probability: float = 0.25,
    full_action_space: bool = False,
    max_episode_steps: Optional[int] = None,
    **kwargs: Any,
) -> gym.Env:
    """
    Create any Atari environment from Arcadic Learning Environment (ALE).

    Args:
        env_id (str): The id of the Atari environment to create.
        obs_type (str): The type of observation to return ("rgb" or "ram").
        frameskip (int): The number of frames to skip between actions.
        repeat_action_probability (float): The probability of repeating the last action.
        full_action_space (bool): Whether to use the full action space.
        max_episode_steps (Optional[int]): Maximum number of steps per episode.
        **kwargs: Additional keyword arguments for environment configuration.

    Returns:
        gym.Env: The created Atari environment.
    """

    return gym.make(
        env_id,
        obs_type=obs_type,
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )


def list_available_atari_envs() -> list[str]:
    """
    Get a list of all available Atari environments in Arcadic Learning Environment (ALE).

    Returns:
        list[str]: List of available Atari environment IDs.
    """

    all_envs = list(gym.envs.registry.keys())
    atari_envs = [env for env in all_envs if env.startswith("ALE/")]
    return sorted(atari_envs)


def list_gymnasium_envs() -> dict[str, list[str]]:
    """
    Get a dictionary of all available Gymnasium environments organized by category.

    Returns:
        dict[str, list[str]]: Dictionary with categories as keys and lists of env IDs as values.
    """
    all_envs = list(gym.envs.registry.keys())

    categories = {
        "classic_control": [],
        "box2d": [],
        "toy_text": [],
        "mujoco": [],
        "atari": [],
        "other": [],
    }

    classic_control_envs = [
        "Acrobot",
        "CartPole",
        "MountainCar",
        "MountainCarContinuous",
        "Pendulum",
    ]
    box2d_envs = ["BipedalWalker", "CarRacing", "LunarLander"]
    toy_text_envs = ["Blackjack", "Taxi", "CliffWalking", "FrozenLake"]
    mujoco_envs = [
        "Ant",
        "HalfCheetah",
        "Hopper",
        "Humanoid",
        "HumanoidStandup",
        "InvertedDoublePendulum",
        "InvertedPendulum",
        "Pusher",
        "Reacher",
        "Swimmer",
        "Walker2d",
    ]

    for env_id in all_envs:
        if env_id.startswith("ALE/"):
            categories["atari"].append(env_id)
        elif any(
            env.startswith(env_id + "-") or env_id.startswith(env + "-")
            for env in classic_control_envs
        ):
            categories["classic_control"].append(env_id)
        elif any(
            env.startswith(env_id + "-") or env_id.startswith(env + "-")
            for env in box2d_envs
        ):
            categories["box2d"].append(env_id)
        elif any(
            env.startswith(env_id + "-") or env_id.startswith(env + "-")
            for env in toy_text_envs
        ):
            categories["toy_text"].append(env_id)
        elif any(
            env.startswith(env_id + "-") or env_id.startswith(env + "-")
            for env in mujoco_envs
        ):
            categories["mujoco"].append(env_id)
        elif env_id not in [
            "Ant-v5",
            "HalfCheetah-v5",
            "Hopper-v5",
            "Humanoid-v5",
            "Swimmer-v5",
            "Walker2d-v5",
            "Reacher-v5",
            "Pusher-v5",
            "InvertedPendulum-v5",
            "InvertedDoublePendulum-v5",
        ]:
            categories["other"].append(env_id)

    for key in categories:
        categories[key] = sorted(categories[key])

    return categories


def get_all_gymnasium_env_ids() -> list[str]:
    """
    Get a flat list of all available Gymnasium environment IDs (excluding Atari).

    Returns:
        list[str]: List of all available Gymnasium environment IDs.
    """
    categories = list_gymnasium_envs()
    all_envs = []
    for category, envs in categories.items():
        if category != "atari":
            all_envs.extend(envs)
    return sorted(all_envs)
