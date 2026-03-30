"""Multi-worker orchestration for distributed generation.

Provides deterministic worker spawning with seed splitting for parallel
episode generation.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process, Queue
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .api import RNGManager


def worker_loop(
    worker_id: int,
    config: Dict[str, Any],
    master_seed: int,
    task_queue: Queue,
    result_queue: Queue,
) -> None:
    """Worker process loop for episode generation."""
    from .envs.basic_env import BasicEnv
    from .exporters.hdf5 import HDF5Exporter

    # Derive deterministic seed for this worker
    rng = RNGManager(master_seed)
    worker_seed = rng.split(f"worker_{worker_id}").backend_seed
    worker_seed = int(worker_seed)

    env = BasicEnv(config)
    exporter = None

    while True:
        try:
            task = task_queue.get(timeout=1.0)
        except Exception:
            continue

        if task is None:
            break

        episode_id = task["episode_id"]
        steps = task["steps"]
        out_path = task.get("out_path")

        # Reset with worker-derived seed
        seed = worker_seed + episode_id
        obs, info = env.reset(seed=seed)

        frames = []
        actions = []
        rewards = []
        dones = []

        for t in range(steps):
            action = []
            obs, reward, done, info = env.step(action)
            frames.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            if done:
                break

        result = {
            "episode_id": episode_id,
            "frames": frames,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "metadata": {"seed": seed, "steps": len(frames)},
        }
        result_queue.put(result)

    env.close()


class MultiWorkerGenerator:
    """Multi-worker deterministic episode generator.

    Args:
        config: Environment config dict.
        num_workers: Number of worker processes.
        master_seed: Master seed for deterministic worker seed derivation.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        num_workers: int = 4,
        master_seed: int = 42,
    ):
        self.config = config
        self.num_workers = num_workers
        self.master_seed = master_seed
        self._workers: List[Process] = []
        self._task_queue: Queue = Queue()
        self._result_queue: Queue = Queue()

    def start(self) -> None:
        """Start worker processes."""
        for i in range(self.num_workers):
            p = Process(
                target=worker_loop,
                args=(
                    i,
                    self.config,
                    self.master_seed,
                    self._task_queue,
                    self._result_queue,
                ),
            )
            p.start()
            self._workers.append(p)

    def generate(
        self,
        num_episodes: int,
        steps_per_episode: int,
        exporter: Optional[Any] = None,
    ) -> int:
        """Queue generation tasks and collect results.

        Args:
            num_episodes: Number of episodes to generate.
            steps_per_episode: Steps per episode.
            exporter: Optional exporter to write results (e.g., HDF5Exporter).

        Returns:
            Number of episodes generated.
        """
        # Enqueue tasks
        for ep_id in range(num_episodes):
            self._task_queue.put({"episode_id": ep_id, "steps": steps_per_episode})

        # Send stop signals
        for _ in range(self.num_workers):
            self._task_queue.put(None)

        # Collect results
        generated = 0
        for _ in range(num_episodes):
            result = self._result_queue.get()
            if exporter:
                exporter.add_episode(
                    frames=result["frames"],
                    actions=result["actions"],
                    rewards=result["rewards"],
                    dones=result["dones"],
                    metadata=result["metadata"],
                )
            generated += 1

        return generated

    def stop(self) -> None:
        """Stop worker processes."""
        for p in self._workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        self._workers.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
