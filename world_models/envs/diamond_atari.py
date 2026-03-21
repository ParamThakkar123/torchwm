import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional


class DiamondAtariWrapper(gym.Wrapper):
    """
    Atari wrapper for DIAMOND following the paper specifications:
    - frameskip: number of frames to skip (default 4)
    - max_noop: maximum number of noop actions at reset (default 30)
    - terminate_on_life_loss: terminate episode when life is lost (default True)
    - reward_clip: clip rewards to [-1, 0, 1] (default True)
    - resize: resize observations to specified size (default 64x64)
    """

    def __init__(
        self,
        env: gym.Env,
        frameskip: int = 4,
        max_noop: int = 30,
        terminate_on_life_loss: bool = True,
        reward_clip: bool = True,
        resize: Optional[Tuple[int, int]] = (64, 64),
    ):
        super().__init__(env)
        self.frameskip = frameskip
        self.max_noop = max_noop
        self.terminate_on_life_loss = terminate_on_life_loss
        self.reward_clip = reward_clip
        self.resize = resize

        self.lives = 0
        self._last_lives = 0

        if resize is not None:
            self._height, self._width = resize
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8
            )

    def _apply_frameskip(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply frameskip by repeating the action."""
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.frameskip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                done = True
                break

            if self.terminate_on_life_loss:
                if hasattr(self.env, "ale") and hasattr(self.env.ale, "lives"):
                    self.lives = self.env.ale.lives()
                    if self.lives < self._last_lives and self.lives > 0:
                        done = True
                        info["life_lost"] = True
                        break

        self._last_lives = self.lives

        if self.reward_clip:
            total_reward = float(np.clip(total_reward, -1, 1))

        return obs, total_reward, done, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self._apply_frameskip(action)

        if self.resize is not None:
            obs = self._resize_obs(obs)

        return obs, reward, done, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)

        if self.resize is not None:
            obs = self._resize_obs(obs)

        if self.terminate_on_life_loss:
            if hasattr(self.env, "ale") and hasattr(self.env.ale, "lives"):
                self.lives = self.env.ale.lives()
            else:
                self.lives = 0
            self._last_lives = self.lives

        noops = np.random.randint(1, self.max_noop + 1)
        for _ in range(noops):
            action = self.env.action_space.sample()
            if action == 0:
                obs, _, done, _ = self.env.step(action)
                if self.resize is not None:
                    obs = self._resize_obs(obs)
                if done:
                    obs, info = self.env.reset(**kwargs)
                    if self.resize is not None:
                        obs = self._resize_obs(obs)
                    break

        return obs, info

    def _resize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Resize observation to target size."""
        if obs.shape[:2] == (self._height, self._width):
            return obs

        import cv2

        obs = cv2.resize(obs, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return obs.astype(np.uint8)


def make_diamond_atari_env(
    game: str,
    frameskip: int = 4,
    max_noop: int = 30,
    terminate_on_life_loss: bool = True,
    reward_clip: bool = True,
    resize: Tuple[int, int] = (64, 64),
    seed: Optional[int] = None,
) -> DiamondAtariWrapper:
    """
    Create a DIAMOND-compatible Atari environment.

    Args:
        game: Atari game name (e.g., "Breakout-v5")
        frameskip: Number of frames to skip between actions
        max_noop: Maximum number of noop actions at reset
        terminate_on_life_loss: Whether to terminate on life loss
        reward_clip: Whether to clip rewards to [-1, 0, 1]
        resize: Target size for observations
        seed: Random seed

    Returns:
        DiamondAtariWrapper: Configured Atari environment
    """
    env = gym.make(
        game,
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    env = DiamondAtariWrapper(
        env=env,
        frameskip=frameskip,
        max_noop=max_noop,
        terminate_on_life_loss=terminate_on_life_loss,
        reward_clip=reward_clip,
        resize=resize,
    )

    return env
