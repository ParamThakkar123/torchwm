from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from multiprocessing import Queue
import numpy as np
import torch
from typing import List, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod


class WorkerError(RuntimeError):
    """Raised when a vectorized environment worker reports or exits with an error."""


class SimWorker(mp.Process):
    """
    Worker process that manages a batch of environment instances.
    Handles batched stepping for parallel rollouts.
    """

    def __init__(
        self,
        worker_id: int,
        env_factory: Callable,
        num_envs: int,
        command_queue: Queue,
        result_queue: Queue,
        seed: Optional[int] = None,
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.env_factory = env_factory
        self.num_envs = num_envs
        self.command_queue = command_queue
        self.result_queue = result_queue
        self.seed = seed
        self.envs: List[Any] = []
        self.dones = [False] * num_envs  # Track done status per env
        self.last_obs: List[Any] = []  # Store last obs for done envs
        self.running = True

    def run(self):
        """Main worker loop."""
        try:
            self._init_envs()
        except Exception as exc:
            self._report_error("init", exc)
            return

        while self.running:
            try:
                command = self.command_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as exc:
                self._report_error("queue_get", exc)
                break

            if command is None:  # Shutdown signal
                break

            try:
                cmd_type, data = command
                if cmd_type == "step":
                    results = self._step_batch(data)
                    self.result_queue.put(("step_result", results))
                elif cmd_type == "reset":
                    results = self._reset_batch()
                    self.result_queue.put(("reset_result", results))
                elif cmd_type == "render":
                    results = self._render_batch()
                    self.result_queue.put(("render_result", results))
                elif cmd_type == "close":
                    self._close_batch()
                    self.result_queue.put(("close_result", None))
                    break
                else:
                    raise ValueError(f"Unknown worker command: {cmd_type!r}")
            except Exception as exc:
                self._report_error(cmd_type if "cmd_type" in locals() else "unknown", exc)
                break

        self._close_batch()

    def _init_envs(self):
        """Initialize worker environments with deterministic per-env seeds."""
        self.envs = []
        self.last_obs = []
        for i in range(self.num_envs):
            env_seed = (
                self.seed + self.worker_id * self.num_envs + i
                if self.seed is not None
                else None
            )
            env = self.env_factory()
            if env_seed is not None and hasattr(env, "seed"):
                env.seed(env_seed)
            self.envs.append(env)
            # Initial obs will be set in reset
            self.last_obs.append(None)

    def _report_error(self, command: str, exc: BaseException):
        """Send structured exception details to the parent before exiting."""
        error = {
            "worker_id": self.worker_id,
            "command": command,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            self.result_queue.put(("worker_error", error))
        except Exception:
            pass

    def _step_batch(self, actions: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Step all environments in batch."""
        results = []
        for i, action in enumerate(actions):
            if self.dones[i]:
                # Env is done, return last obs with done=True
                results.append(
                    {"obs": self.last_obs[i], "reward": 0.0, "done": True, "info": {}}
                )
            else:
                obs, reward, done, info = self.envs[i].step(action)
                self.dones[i] = bool(done)
                if done:
                    # Store last obs for future steps
                    self.last_obs[i] = obs
                results.append(
                    {"obs": obs, "reward": reward, "done": done, "info": info}
                )
        return results

    def _reset_batch(self) -> List[Dict[str, Any]]:
        """Reset all environments."""
        results = []
        for i, env in enumerate(self.envs):
            obs = env.reset()
            self.dones[i] = False
            self.last_obs[i] = obs
            results.append({"obs": obs})
        return results

    def _render_batch(self) -> List[np.ndarray]:
        """Render all environments."""
        results = []
        for env in self.envs:
            frame = env.render()
            results.append(frame)
        return results

    def _close_batch(self):
        """Close all environments."""
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()


class VectorizedEnv(ABC):
    """
    Abstract base class for vectorized environments.
    Manages multiple worker processes for parallel simulation.
    """

    def __init__(
        self,
        env_factory: Callable,
        num_workers: int = 2,
        envs_per_worker: int = 4,
        seed: Optional[int] = None,
    ):
        self.env_factory = env_factory
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.total_envs = num_workers * envs_per_worker
        self.seed = seed

        # Create communication queues
        self.command_queues: List[Queue] = [Queue() for _ in range(num_workers)]
        self.result_queues: List[Queue] = [Queue() for _ in range(num_workers)]

        # Start workers
        for i in range(num_workers):
            self._start_worker(i)

        # Cache observation and action spaces from a dummy env
        dummy_env = env_factory()
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        if hasattr(dummy_env, "close"):
            dummy_env.close()

    def _start_worker(self, worker_id: int):
        """Start one worker process and store it in worker-id order."""
        worker = SimWorker(
            worker_id=worker_id,
            env_factory=self.env_factory,
            num_envs=self.envs_per_worker,
            command_queue=self.command_queues[worker_id],
            result_queue=self.result_queues[worker_id],
            seed=self.seed,
        )
        worker.start()
        if not hasattr(self, "workers"):
            self.workers = [None] * self.num_workers
        self.workers[worker_id] = worker

    def _restart_worker(self, worker_id: int):
        """Restart a failed worker process so future calls can recover."""
        worker = self.workers[worker_id]
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=5.0)
        self._start_worker(worker_id)

    def _raise_worker_error(self, error: Dict[str, Any]):
        """Restart the failed process and raise a parent-side exception."""
        worker_id = error["worker_id"]
        self._restart_worker(worker_id)
        message = (
            f"Worker {worker_id} failed during {error['command']}: "
            f"{error['error_type']}: {error['message']}\n"
            f"{error['traceback']}"
        )
        raise WorkerError(message)

    def _collect_worker_results(self, expected_result: str) -> List[Any]:
        """Collect one result per worker, propagating failures with tracebacks."""
        results_by_worker = []
        for worker_id, result_queue in enumerate(self.result_queues):
            try:
                cmd, data = result_queue.get(timeout=30.0)
            except queue.Empty as exc:
                if not self.workers[worker_id].is_alive():
                    exitcode = self.workers[worker_id].exitcode
                    self._restart_worker(worker_id)
                    raise WorkerError(
                        f"Worker {worker_id} exited with code "
                        f"{exitcode} while waiting for "
                        f"{expected_result}."
                    ) from exc
                raise WorkerError(
                    f"Timed out waiting for worker {worker_id} to return "
                    f"{expected_result}."
                ) from exc

            if cmd == "worker_error":
                self._raise_worker_error(data)
            if cmd != expected_result:
                raise WorkerError(
                    f"Worker {worker_id} returned {cmd!r}; expected "
                    f"{expected_result!r}."
                )
            results_by_worker.append(data)
        return results_by_worker

    @abstractmethod
    def step_batch(self, actions: torch.Tensor) -> Dict[str, Any]:
        """Step all environments with batched actions."""
        pass

    @abstractmethod
    def reset_batch(self) -> Dict[str, Any]:
        """Reset all environments."""
        pass

    def render_batch(self) -> List[np.ndarray]:
        """Render all environments."""
        for q in self.command_queues:
            q.put(("render", None))

        results = []
        for data in self._collect_worker_results("render_result"):
            results.extend(data)
        return results

    def close(self):
        """Shutdown all workers."""
        for q in self.command_queues:
            q.put(("close", None))

        for worker in self.workers:
            worker.join(timeout=5.0)

        for q in self.command_queues + self.result_queues:
            q.close()


class TorchVectorizedEnv(VectorizedEnv):
    """
    TorchWM-compatible vectorized environment.
    Returns batched tensors suitable for PyTorch training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wait_for_workers()

    def _wait_for_workers(self):
        """Ensure workers are ready by sending a dummy command."""
        # This is a simple way to sync; could be improved
        pass

    def step_batch(self, actions: torch.Tensor) -> Dict[str, Any]:
        """
        Step all environments with batched actions.

        Args:
            actions: Tensor of shape (total_envs, action_dim)

        Returns:
            Dict with 'obs', 'reward', 'done', 'info' tensors
        """
        assert actions.shape[0] == self.total_envs

        # Split actions by worker
        action_chunks = torch.chunk(actions, self.num_workers, dim=0)

        # Send commands
        for i, chunk in enumerate(action_chunks):
            self.command_queues[i].put(("step", chunk.numpy()))

        # Collect results
        all_obs = []
        all_rewards = []
        all_dones = []
        all_infos = []

        for results in self._collect_worker_results("step_result"):
            for result in results:
                all_obs.append(result["obs"])
                all_rewards.append(result["reward"])
                all_dones.append(result["done"])
                all_infos.append(result["info"])

        # Stack into tensors (assuming dict obs with 'image' key for simplicity)
        # This assumes all obs are dicts with 'image' key of shape (C, H, W)
        obs_images = [obs["image"] for obs in all_obs]
        obs_tensor = torch.stack(
            [torch.from_numpy(img).float() / 255.0 for img in obs_images]
        )

        reward_tensor = torch.tensor(all_rewards, dtype=torch.float32)
        done_tensor = torch.tensor(all_dones, dtype=torch.bool)

        # Return a mapping where obs is nested dict and info remains a list
        return {
            "obs": {"image": obs_tensor},
            "reward": reward_tensor,
            "done": done_tensor,
            "info": all_infos,  # Keep as list for now
        }

    def reset_batch(self) -> Dict[str, Any]:
        """Reset all environments and return initial observations."""
        for q in self.command_queues:
            q.put(("reset", None))

        all_obs = []
        for results in self._collect_worker_results("reset_result"):
            for result in results:
                all_obs.append(result["obs"])

        obs_images = [obs["image"] for obs in all_obs]
        obs_tensor = torch.stack(
            [torch.from_numpy(img).float() / 255.0 for img in obs_images]
        )

        return {"obs": {"image": obs_tensor}}
