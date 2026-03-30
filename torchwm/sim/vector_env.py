"""Vectorized environment wrapper with tensor-first API.

Provides batched environment stepping with torch tensor outputs for efficient
integration with PyTorch training pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None

from .api import BaseEnv


class VectorEnv:
    """Vectorized environment wrapper with tensor-first API.

    Wraps multiple BaseEnv instances and provides batched operations.
    Observations are returned as stacked tensors for direct use in PyTorch.

    Args:
        envs: list of BaseEnv instances.
        stack_dim: dimension to stack observations (default 0 for batch).
    """

    def __init__(self, envs: Sequence[BaseEnv], stack_dim: int = 0):
        self.envs = list(envs)
        self.stack_dim = stack_dim

    def reset(
        self, seeds: Optional[Sequence[Optional[int]]] = None
    ) -> Tuple[Any, Dict]:
        """Reset all environments.

        Args:
            seeds: optional sequence of seeds (one per env).

        Returns:
            observations: batched tensor or list of observations
            info: dict with per-env info
        """
        obs_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            seed = seeds[i] if seeds is not None else None
            obs, info = env.reset(seed=seed)
            obs_list.append(obs)
            info_list.append(info)

        batched_obs = self._stack_obs(obs_list)
        return batched_obs, {"infos": info_list}

    def step(
        self, actions: Sequence[Any]
    ) -> Tuple[Any, List[float], List[bool], List[Dict]]:
        """Step all environments with corresponding actions.

        Args:
            actions: sequence of actions (one per env).

        Returns:
            batched observations, rewards, dones, infos
        """
        obs_list = []
        rew_list = []
        done_list = []
        info_list = []

        for env, act in zip(self.envs, actions):
            obs, reward, done, info = env.step(act)
            obs_list.append(obs)
            rew_list.append(float(reward))
            done_list.append(bool(done))
            info_list.append(info)

        batched_obs = self._stack_obs(obs_list)
        return batched_obs, rew_list, done_list, info_list

    def _stack_obs(self, obs_list: List[Any]) -> Any:
        """Stack observations into batched tensor or list."""
        if torch is None:
            return obs_list

        # Try to convert to torch tensor
        try:
            # Assume obs is numpy array
            tensors = [torch.from_numpy(np.asarray(o)) for o in obs_list]
            return torch.stack(tensors, dim=self.stack_dim)
        except Exception:
            return obs_list

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def __len__(self) -> int:
        return len(self.envs)


class DeterministicVectorEnv(VectorEnv):
    """VectorEnv with deterministic seed assignment.

    Given a master seed, derives deterministic seeds for each worker using
    a simple split (seed + offset). This ensures reproducibility in multi-worker
    generation.
    """

    def __init__(self, envs: Sequence[BaseEnv], master_seed: int, stack_dim: int = 0):
        super().__init__(envs, stack_dim)
        self.master_seed = master_seed

    def reset(
        self, seeds: Optional[Sequence[Optional[int]]] = None
    ) -> Tuple[Any, Dict]:
        if seeds is None:
            seeds = [self.master_seed + i for i in range(len(self.envs))]
        return super().reset(seeds)
