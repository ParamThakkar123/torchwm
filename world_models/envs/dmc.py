import gymnasium as gym
import numpy as np


_MUJOCO_ABI_MISSING_FIELDS = (
    "flex_bandwidth",
    "qLDiagSqrtInv",
    "flex_xvert0",
    "jnt_actgravcomp",
    "solver_iter",
    "tex_rgb",
)


def is_dm_control_mujoco_abi_error(exc: AttributeError) -> bool:
    """Return True for common dm-control/MuJoCo struct-field mismatches."""

    message = str(exc)
    return (
        "MjModel" in message
        or "MjData" in message
        or any(field in message for field in _MUJOCO_ABI_MISSING_FIELDS)
    )


def dm_control_mujoco_abi_message(original_error: AttributeError) -> str:
    """Build a user-facing message for mismatched dm-control/MuJoCo wheels."""

    return (
        "DeepMind Control failed while initializing MuJoCo. This usually means "
        "the installed dm-control and mujoco wheels are ABI-incompatible. "
        "Reinstall a matching pair with `pip install --upgrade --force-reinstall "
        "'dm-control>=1.0.28' 'mujoco>=3.3.1'`, restart the Python kernel, "
        "and verify with `pip show dm-control mujoco`. Original error: "
        f"{original_error}"
    )


class DeepMindControlEnv:
    """Gym-style adapter for DeepMind Control Suite tasks.

    The wrapper exposes DMC observations and actions through Gym spaces and
    adds a rendered RGB image to each observation dict so image-based world
    model pipelines can train consistently across backends.

    Features:
        - Parses domain-task names (e.g., "cheetah-run" -> domain="cheetah", task="run")
        - Automatically handles special cases like "cup" -> "ball_in_cup"
        - Renders RGB images at configurable resolution
        - Returns observations as dict with both state vectors and images

    Args:
        name (str): Environment name in format "domain-task" (e.g., "cheetah-run").
        seed (int): Random seed for environment initialization.
        size (tuple): Target image size as (height, width) (default: (64, 64)).
        camera (int, optional): Camera ID for rendering. Defaults to 0 for most
            domains, 2 for quadruped.

    Attributes:
        observation_space (gym.spaces.Dict): Dict space with state keys and "image".
        action_space (gym.spaces.Box): Continuous action space from DMC spec.

    Example:
        >>> env = DeepMindControlEnv("cheetah-run", seed=0, size=(64, 64))
        >>> obs = env.reset()
        >>> print(obs.keys())  # dict_keys(['position', 'velocity', 'image'])
    """

    def __init__(self, name, seed, size=(64, 64), camera=None):
        domain, task = name.split("-", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            from dm_control import suite

            try:
                self._env = suite.load(domain, task, task_kwargs={"random": seed})
            except AttributeError as exc:
                if is_dm_control_mujoco_abi_error(exc):
                    raise RuntimeError(dm_control_mujoco_abi_message(exc)) from exc
                raise
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs["image"] = self.render().transpose(2, 0, 1).copy()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs["image"] = self.render().transpose(2, 0, 1).copy()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
