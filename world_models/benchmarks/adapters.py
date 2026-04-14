from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from world_models.training.train_diamond import DiamondAgent
from world_models.training.train_iris import IRISTrainer


class BaseAdapter:
    def __init__(self, env_spec: Any | None = None, seed: int = 0, **kwargs):
        self.env_spec = env_spec
        self.seed = seed

    def load_checkpoint(self, path: str):
        raise NotImplementedError

    def evaluate(self, num_episodes: int = 1, render: bool = False):
        """Return standardized output. Preferred format: dict with key 'episode_returns' -> List[float]"""
        raise NotImplementedError


class DiamondAdapter(BaseAdapter):
    def __init__(self, env_spec: Any | None = None, seed: int = 0, **kwargs):
        super().__init__(env_spec, seed)
        game = (
            env_spec.get("game")
            if isinstance(env_spec, dict)
            else getattr(env_spec, "game", None)
        )
        self.agent = DiamondAgent(
            DiamondAgent.__init__.__annotations__.get("config") or {}
        )
        # The above is a minimal placeholder; create config via DiamondConfig in runner if needed
        # To avoid circular imports/config complexities, defer user to pass full adapter kwargs.

    def load_checkpoint(self, path: str):
        try:
            self.agent.load_checkpoint(path)
        except Exception:
            raise

    def evaluate(self, num_episodes: int = 1, render: bool = False):
        # DiamondAgent.evaluate returns a float mean reward per episode by default
        avg = self.agent.evaluate(num_episodes=num_episodes)
        return {"episode_returns": [float(avg)]}


class IRISAdapter(BaseAdapter):
    def __init__(self, env_spec: Any | None = None, seed: int = 0, **kwargs):
        super().__init__(env_spec, seed)
        game = None
        if isinstance(env_spec, dict):
            game = env_spec.get("game")
        self.trainer = IRISTrainer(game=game or "ALE/Pong-v5", seed=seed)

    def load_checkpoint(self, path: str):
        # IRIS agent provides save/load on agent; delegate if trainer has agent
        try:
            self.trainer.agent.load(path)
        except Exception:
            raise

    def evaluate(self, num_episodes: int = 1, render: bool = False):
        res = self.trainer.evaluate(num_episodes=num_episodes, render=render)
        if render:
            # (episode_returns_array, videos_list, latents_array)
            ep_returns, videos, latents = res
            return {
                "episode_returns": list(ep_returns),
                "videos": videos,
                "latents": latents,
            }
        else:
            # trainer.evaluate returns a dict with summary keys
            # We convert to repeated mean for compatibility
            if isinstance(res, dict) and "eval_mean_return" in res:
                return {
                    "episode_returns": [
                        float(res["eval_mean_return"]) for _ in range(num_episodes)
                    ]
                }
            return {"episode_returns": []}
