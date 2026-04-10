import multiprocessing as mp
from multiprocessing import Queue
import numpy as np
import torch
from typing import List, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod
import threading
import asyncio
import inspect
import importlib.util

# Conditional imports
if importlib.util.find_spec("jax"):
    import jax
else:
    jax = None

try:
    from world_models.cuda_kernels import batched_normalize

    HAS_CUDA_KERNELS = True
except ImportError:
    HAS_CUDA_KERNELS = False


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
        # Initialize environments
        self.envs = []
        self.last_obs = []
        for i in range(self.num_envs):
            env_seed = (
                self.seed + self.worker_id * self.num_envs + i if self.seed else None
            )
            env = self.env_factory()
            if env_seed is not None and hasattr(env, "seed"):
                env.seed(env_seed)
            self.envs.append(env)
            # Initial obs will be set in reset
            self.last_obs.append(None)

        while self.running:
            try:
                command = self.command_queue.get(timeout=1.0)
                if command is None:  # Shutdown signal
                    break

                cmd_type, data = command
                if cmd_type == "step":
                    actions = data
                    results = self._step_batch(actions)
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
            except Exception:
                continue  # Timeout or error, keep running

        self._close_batch()

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
        self.command_queues = [Queue() for _ in range(num_workers)]
        self.result_queues = [Queue() for _ in range(num_workers)]

        # Start workers
        self.workers = []
        for i in range(num_workers):
            worker = SimWorker(
                worker_id=i,
                env_factory=env_factory,
                num_envs=envs_per_worker,
                command_queue=self.command_queues[i],
                result_queue=self.result_queues[i],
                seed=seed,
            )
            worker.start()
            self.workers.append(worker)

        # Cache observation and action spaces from a dummy env
        dummy_env = env_factory()
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        if hasattr(dummy_env, "close"):
            dummy_env.close()

    @abstractmethod
    def step_batch(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Step all environments with batched actions."""
        pass

    @abstractmethod
    def reset_batch(self) -> Dict[str, torch.Tensor]:
        """Reset all environments."""
        pass

    def render_batch(self) -> List[np.ndarray]:
        """Render all environments."""
        for q in self.command_queues:
            q.put(("render", None))

        results = []
        for q in self.result_queues:
            cmd, data = q.get()
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

    def step_batch(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        for q in self.result_queues:
            cmd, results = q.get()
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

        return {
            "obs": {"image": obs_tensor},
            "reward": reward_tensor,
            "done": done_tensor,
            "info": all_infos,  # Keep as list for now
        }

    def reset_batch(self) -> Dict[str, torch.Tensor]:
        """Reset all environments and return initial observations."""
        for q in self.command_queues:
            q.put(("reset", None))

        all_obs = []
        for q in self.result_queues:
            cmd, results = q.get()
            for result in results:
                all_obs.append(result["obs"])

        obs_images = [obs["image"] for obs in all_obs]
        obs_tensor = torch.stack(
            [torch.from_numpy(img).float() / 255.0 for img in obs_images]
        )

        return {"obs": {"image": obs_tensor}}


class GPUVectorizedEnv(VectorizedEnv):
    """
    GPU-accelerated vectorized environment for environments like Isaac Lab or Brax.
    Supports both single environments (like Brax) and natively vectorized GPU environments (like Isaac Lab).
    Uses CUDA streams for async batching and parallel simulation on GPU.
    Compatible with PyTorch and JAX-based envs via automatic array conversion.
    """

    def __init__(
        self,
        env_factory: Callable,
        num_envs: int = 32,
        device: torch.device = None,
        seed: Optional[int] = None,
        async_batching: bool = True,
    ):
        self.env_factory = env_factory
        self.num_envs = num_envs
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = seed
        self.async_batching = async_batching

        # Try to create a vectorized environment (e.g., Isaac Lab)
        try:
            sig = inspect.signature(env_factory)
            if "num_envs" in sig.parameters:
                self.env = env_factory(num_envs=num_envs, device=self.device, seed=seed)
                self.is_vectorized = True
                self.envs = [self.env]  # For compatibility
            else:
                # Try calling with num_envs anyway
                self.env = env_factory(num_envs=num_envs)
                self.is_vectorized = True
                self.envs = [self.env]
        except (TypeError, AttributeError):
            # Fallback to multiple single environments
            self.envs = []
            for i in range(num_envs):
                env_seed = seed + i if seed else None
                env = env_factory()
                if env_seed is not None and hasattr(env, "seed"):
                    env.seed(env_seed)
                self.envs.append(env)
            self.env = None
            self.is_vectorized = False

        # Apply JIT compilation for Brax envs if available
        self._apply_jit_if_brax()

        # Cache spaces
        dummy_env = self.envs[0] if not self.is_vectorized else self.env
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        if hasattr(dummy_env, "close") and not self.is_vectorized:
            dummy_env.close()  # Don't close if vectorized

        # CUDA streams for async operations
        if self.device.type == "cuda" and self.async_batching:
            self.stream = torch.cuda.current_stream(self.device)
            self.compute_stream = torch.cuda.Stream(self.device)
        else:
            self.stream = None
            self.compute_stream = None

        # For async batching
        self.action_buffer = None
        self.obs_buffer = None

        # Observation wrapper
        self.obs_wrapper = IsaacLabObsWrapper(num_envs=num_envs, device=self.device)

    def _apply_jit_if_brax(self):
        """Apply JAX JIT compilation to Brax env methods for better performance."""
        if not self.is_vectorized or self.env is None or jax is None:
            return

        # Check if this is a Brax environment
        is_brax = False
        # Brax envs typically have a 'sys' attribute or are from brax.envs
        if hasattr(self.env, "sys") or str(type(self.env)).startswith(
            "<class 'brax.envs"
        ):
            is_brax = True

        if is_brax:
            try:
                # JIT compile the step and reset methods
                self.env.step = jax.jit(self.env.step)
                self.env.reset = jax.jit(self.env.reset)
                print(
                    "Applied JAX JIT compilation to Brax environment for improved performance."
                )
            except Exception as e:
                print(f"Warning: Failed to apply JIT to Brax env: {e}")

    def step_batch(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Step all environments with batched actions on GPU.

        Args:
            actions: Tensor of shape (num_envs, action_dim) on device

        Returns:
            Dict with 'obs', 'reward', 'done' tensors
        """
        assert actions.shape[0] == self.num_envs
        assert actions.device == self.device

        if self.async_batching and self.compute_stream is not None:
            # Async batching: overlap computation with data transfer
            with torch.cuda.stream(self.compute_stream):
                return self._step_batch_async(actions)
        else:
            return self._step_batch_sync(actions)

    def _step_batch_sync(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Synchronous batch stepping."""
        if self.is_vectorized:
            # Vectorized environment (e.g., Isaac Lab, Brax)
            actions_np = actions.detach().cpu().numpy()
            obs, reward, terminated, truncated, info = self.env.step(actions_np)

            # Convert JAX/PyTorch arrays to numpy if needed
            obs = (
                {k: np.asarray(v) for k, v in obs.items()}
                if isinstance(obs, dict)
                else np.asarray(obs)
            )
            reward = np.asarray(reward)
            terminated = np.asarray(terminated)
            truncated = np.asarray(truncated)

            # Convert to TorchWM format
            obs_dict = self.obs_wrapper(obs)
            reward_tensor = torch.from_numpy(reward).float().to(self.device)
            terminated_tensor = torch.from_numpy(terminated).bool().to(self.device)
            truncated_tensor = torch.from_numpy(truncated).bool().to(self.device)
            done_tensor = terminated_tensor | truncated_tensor

            return {
                "obs": obs_dict,
                "reward": reward_tensor,
                "done": done_tensor,
                "info": info,
            }
        else:
            # Multiple single environments
            actions_cpu = actions.cpu().numpy()

            all_obs = []
            all_rewards = []
            all_dones = []

            for i, env in enumerate(self.envs):
                action = actions_cpu[i]
                obs, reward, done, info = env.step(action)
                all_obs.append(obs["image"] if isinstance(obs, dict) else obs)
                all_rewards.append(reward)
                all_dones.append(done)

            # Stack and move to GPU
            obs_tensor = torch.stack(
                [torch.from_numpy(o).float() / 255.0 for o in all_obs]
            ).to(self.device)
            reward_tensor = torch.tensor(
                all_rewards, dtype=torch.float32, device=self.device
            )
            done_tensor = torch.tensor(all_dones, dtype=torch.bool, device=self.device)

            return {
                "obs": {"image": obs_tensor},
                "reward": reward_tensor,
                "done": done_tensor,
                "info": [{}] * self.num_envs,  # Placeholder
            }

    def _step_batch_async(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Asynchronous batch stepping with CUDA streams for overlapping compute and transfer."""
        if not self.is_vectorized:
            # For multiple envs, async doesn't help much, fallback to sync
            return self._step_batch_sync(actions)

        # For vectorized envs, use CUDA streams to overlap data transfer and computation
        if self.device.type == "cuda" and self.compute_stream is not None:
            # Record event before data transfer
            transfer_start = torch.cuda.Event()
            transfer_start.record()

            # Transfer actions to CPU asynchronously (if needed)
            actions_np = actions.detach().cpu().numpy()  # This is sync, but small

            # Switch to compute stream
            with torch.cuda.stream(self.compute_stream):
                # Simulate async env step (in real GPU env, this would be GPU ops)
                # For Isaac Lab, step is CPU numpy, so limited async benefit
                obs, reward, terminated, truncated, info = self.env.step(actions_np)

                # Convert to TorchWM format asynchronously on GPU
                obs_dict = self.obs_wrapper(obs)
                reward_tensor = (
                    torch.from_numpy(reward).float().to(self.device, non_blocking=True)
                )
                terminated_tensor = (
                    torch.from_numpy(terminated)
                    .bool()
                    .to(self.device, non_blocking=True)
                )
                truncated_tensor = (
                    torch.from_numpy(truncated)
                    .bool()
                    .to(self.device, non_blocking=True)
                )
                done_tensor = terminated_tensor | truncated_tensor

                result = {
                    "obs": obs_dict,
                    "reward": reward_tensor,
                    "done": done_tensor,
                    "info": info,
                }

            # Record event after computation
            compute_end = torch.cuda.Event()
            compute_end.record(self.compute_stream)

            # Wait for computation to finish before returning
            compute_end.synchronize()

            return result
        else:
            return self._step_batch_sync(actions)

    def reset_batch(self) -> Dict[str, torch.Tensor]:
        """Reset all environments on GPU."""
        if self.is_vectorized:
            obs, info = self.env.reset()
            # Convert JAX/PyTorch arrays to numpy if needed
            obs = (
                {k: np.asarray(v) for k, v in obs.items()}
                if isinstance(obs, dict)
                else np.asarray(obs)
            )
            obs_dict = self.obs_wrapper(obs)
            return {"obs": obs_dict}
        else:
            all_obs = []
            for env in self.envs:
                obs = env.reset()
                all_obs.append(obs["image"] if isinstance(obs, dict) else obs)

            obs_tensor = torch.stack(
                [torch.from_numpy(o).float() / 255.0 for o in all_obs]
            ).to(self.device)
            return {"obs": {"image": obs_tensor}}

    def render_batch(self) -> List[np.ndarray]:
        """Render all environments (may not be GPU-accelerated)."""
        if self.is_vectorized:
            # For vectorized, may not have batched render
            frames = []
            for i in range(self.num_envs):
                frame = self.env.render()
                frames.append(frame)
            return frames
        else:
            results = []
            for env in self.envs:
                frame = env.render()
                results.append(frame)
            return results

    def close(self):
        """Close all environments."""
        if self.is_vectorized:
            if hasattr(self.env, "close"):
                self.env.close()
        else:
            for env in self.envs:
                if hasattr(env, "close"):
                    env.close()


class IsaacLabObsWrapper:
    """
    Wrapper to convert Isaac Lab and Brax observations to TorchWM format.
    Handles PyTorch, NumPy, and JAX arrays automatically.
    Assumes observations are dicts with image-like data.
    """

    def __init__(self, num_envs: int = 32, device: torch.device = None):
        self.num_envs = num_envs
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._size = (64, 64)  # Default TorchWM size

    def __call__(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert Isaac Lab/Brax obs dict to TorchWM obs dict."""
        if isinstance(obs, dict) and "image" in obs:
            # Already in format
            img = obs["image"]
            # Convert JAX arrays to numpy if needed
            if hasattr(img, "__array__") and not isinstance(img, np.ndarray):
                img = np.asarray(img)
            img = torch.from_numpy(img).float() / 255.0
            if img.dim() == 4:  # (N, C, H, W)
                img = batched_normalize(img) if HAS_CUDA_KERNELS else img
                return {"image": img.to(self.device)}
            elif img.dim() == 3:  # Single (C, H, W)
                img = (
                    batched_normalize(img.unsqueeze(0)).squeeze(0)
                    if HAS_CUDA_KERNELS
                    else img
                )
                return {"image": img.unsqueeze(0).to(self.device)}
        elif isinstance(obs, dict):
            # Try to find image-like data
            for key, value in obs.items():
                if isinstance(value, (np.ndarray,)) or hasattr(value, "__array__"):
                    # Convert JAX to numpy
                    if hasattr(value, "__array__") and not isinstance(
                        value, np.ndarray
                    ):
                        value = np.asarray(value)
                    value = np.asarray(value)
                    if value.ndim in (3, 4):
                        img = torch.from_numpy(value).float()
                        if img.shape[-1] in (1, 3):  # HWC
                            img = (
                                img.permute(0, 3, 1, 2)
                                if img.dim() == 4
                                else img.permute(2, 0, 1)
                            )
                        if img.dim() == 3:
                            img = img.unsqueeze(0)
                        return {"image": (img / 255.0).to(self.device)}

        # Fallback: synthesize image from vector obs
        if isinstance(obs, (np.ndarray,)) or hasattr(obs, "__array__"):
            if hasattr(obs, "__array__") and not isinstance(obs, np.ndarray):
                obs = np.asarray(obs)
            obs = np.asarray(obs)
            if obs.ndim == 2:  # (N, D)
                batch_size = obs.shape[0]
                images = []
                for i in range(batch_size):
                    vec = obs[i]
                    image = self._vector_to_image(vec)
                    images.append(image)
                img_tensor = torch.stack(images)
                return {"image": img_tensor.to(self.device)}

        # If all else fails, return zeros
        return {"image": torch.zeros(self.num_envs, 3, *self._size, device=self.device)}

    def _vector_to_image(self, vector: np.ndarray) -> torch.Tensor:
        """Convert vector observation to synthetic image."""
        vec = torch.from_numpy(vector).float()
        if vec.numel() == 0:
            return torch.zeros(3, self._size[0], self._size[1])

        vmin, vmax = vec.min(), vec.max()
        if vmax > vmin:
            vec = (vec - vmin) / (vmax - vmin)

        image = torch.zeros(3, self._size[0], self._size[1])
        bands = min(8, vec.numel())
        band_w = max(1, self._size[1] // max(1, bands))
        for i in range(bands):
            start = i * band_w
            end = min(self._size[1], start + band_w)
            image[:, :, start:end] = vec[i].clamp(0, 1)
        return image
