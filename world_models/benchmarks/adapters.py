from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from world_models.configs.diamond_config import DiamondConfig
from world_models.configs.iris_config import IRISConfig
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
        # env_spec can be a dict with keys like 'game', or a simple string game name
        if isinstance(env_spec, dict):
            game = env_spec.get("game", None)
        elif isinstance(env_spec, str):
            game = env_spec
        else:
            game = getattr(env_spec, "game", None)

        preset = kwargs.get("preset", None)
        device = kwargs.get("device", "cuda")

        cfg = DiamondConfig(preset=preset)
        if game:
            cfg.game = game
        cfg.seed = seed
        cfg.device = device

        self.agent = DiamondAgent(cfg)

    def load_checkpoint(self, path: str):
        try:
            self.agent.load_checkpoint(path)
        except Exception:
            raise

    def evaluate(self, num_episodes: int = 1, render: bool = False):
        # DiamondAgent.evaluate returns a float mean reward per episode by default.
        # To produce per-episode returns we run single-episode evaluations repeatedly.
        episode_returns: List[float] = []
        for _ in range(num_episodes):
            try:
                r = self.agent.evaluate(num_episodes=1)
                # If evaluate returns an average scalar, treat as single-episode reward
                if isinstance(r, (int, float)):
                    episode_returns.append(float(r))
                elif isinstance(r, dict) and "episode_returns" in r:
                    vals = r["episode_returns"]
                    if isinstance(vals, (list, tuple)) and len(vals) > 0:
                        episode_returns.extend([float(v) for v in vals])
                    else:
                        # fallback: use mean
                        episode_returns.append(float(np.mean(vals) if vals else 0.0))
                else:
                    # Unknown format: try to coerce
                    try:
                        episode_returns.append(float(r))
                    except Exception:
                        episode_returns.append(0.0)
            except Exception:
                # If per-episode evaluation fails, append zero as safe fallback
                episode_returns.append(0.0)

        return {"episode_returns": episode_returns}


class IRISAdapter(BaseAdapter):
    def __init__(self, env_spec: Any | None = None, seed: int = 0, **kwargs):
        super().__init__(env_spec, seed)
        game = None
        if isinstance(env_spec, dict):
            game = env_spec.get("game")
        elif isinstance(env_spec, str):
            game = env_spec

        device = kwargs.get("device", "cuda")
        config = kwargs.get("config", None)
        if config is None:
            cfg = IRISConfig()
        else:
            cfg = config

        # IRISTrainer accepts (game, device, seed, config)
        self.trainer = IRISTrainer(
            game=game or cfg.env, device=device, seed=seed, config=cfg
        )

    def load_checkpoint(self, path: str):
        # IRIS agent provides save/load on agent; delegate if trainer has agent
        try:
            # IRISAgent implements load(path)
            if hasattr(self.trainer, "agent") and hasattr(self.trainer.agent, "load"):
                self.trainer.agent.load(path)
            else:
                raise AttributeError("IRIS agent does not expose load(path)")
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
