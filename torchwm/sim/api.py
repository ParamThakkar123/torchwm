"""Public API signatures for the torchwm simulator package.

This module contains abstract/base interfaces and lightweight adapters used by
the rest of the simulator.  Implementations for adapters (PyBullet, renderers,
exporters) live in adapter modules under torchwm/sim/.

The goal here is to provide clear, well-documented signatures so the
implementation work can proceed against a stable contract.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Iterable
import hashlib
import random
import pickle
import base64

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None
from dataclasses import dataclass

Observation = Any
Action = Any
Info = Dict[str, Any]
Snapshot = Any


class BaseEnv:
    """Abstract environment interface.

    Intentionally similar to Gymnasium's env API but kept minimal and
    backend-agnostic. Implementations must be deterministic when given the
    same RNG streams and configuration.
    """

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Observation, Info]:
        """Reset the environment state.

        Args:
            seed: Optional integer seed for deterministic resets. Implementations
                should derive per-component RNGs from this value using the
                RNGManager contract.
            options: Backend-specific options passed to reset.

        Returns:
            observation, info
        """
        raise NotImplementedError

    def step(self, action: Action) -> Tuple[Observation, float, bool, Info]:
        """Advance the environment by one step using `action`.

        Returns a tuple (observation, reward, done, info).
        """
        raise NotImplementedError

    def render(self, mode: str = "rgb_array") -> Any:
        """Render a frame or return an array representing the current view.

        Typical modes: 'rgb_array' (HWC uint8/float), 'pil' etc. Keep headless
        operation as the primary path; render may be a no-op in headless mode.
        """
        raise NotImplementedError

    def snapshot(self) -> Snapshot:
        """Return a deterministic serializable snapshot of the full world state,
        including any RNG state necessary to resume deterministic execution.
        """
        raise NotImplementedError

    def restore(self, snapshot: Snapshot) -> None:
        """Restore a previously-captured snapshot produced by `snapshot()`."""
        raise NotImplementedError

    def close(self) -> None:
        """Clean up resources (renderer, physics backend processes, files)."""
        raise NotImplementedError


@dataclass
class RNGStreams:
    """Container for PRNG streams derived from a single seed.

    Fields are optional because backends may not require all of them.
    """

    python_random_state: Any = None
    numpy_rng: Any = None
    torch_generator: Any = None
    backend_seed: Optional[int] = None
    # Optionally keep the integer seeds used to construct the streams
    _seeds: Optional[Dict[str, int]] = None

    def to_state(self) -> Dict[str, Any]:
        """Serialize RNGStreams to a JSON-friendly dict using pickle+base64
        for RNG internals where necessary. Returns a mapping that can be
        restored via `RNGStreams.from_state`.
        """
        state: Dict[str, Any] = {
            "backend_seed": int(self.backend_seed)
            if self.backend_seed is not None
            else None
        }

        # python.random state
        if self.python_random_state is not None:
            try:
                py_state = self.python_random_state.getstate()
                state["python_state"] = base64.b64encode(pickle.dumps(py_state)).decode(
                    "ascii"
                )
            except Exception:
                state["python_state"] = None
        else:
            state["python_state"] = None

        # numpy state
        if self.numpy_rng is not None:
            try:
                np_state = self.numpy_rng.bit_generator.state
                state["numpy_state"] = base64.b64encode(pickle.dumps(np_state)).decode(
                    "ascii"
                )
            except Exception:
                state["numpy_state"] = None
        else:
            state["numpy_state"] = None

        # torch generator: store seed if available
        if self.torch_generator is not None and torch is not None:
            try:
                # Torch Generator doesn't expose a stable full state API across versions,
                # so we store a seed if possible.
                seed = None
                try:
                    seed = int(self.torch_generator.initial_seed())
                except Exception:
                    # fallback: use backend_seed
                    seed = (
                        int(self.backend_seed)
                        if self.backend_seed is not None
                        else None
                    )
                state["torch_seed"] = seed
            except Exception:
                state["torch_seed"] = None
        else:
            state["torch_seed"] = None

        # include seeds if present
        if self._seeds is not None:
            state["_seeds"] = dict(self._seeds)
        else:
            state["_seeds"] = None

        return state

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "RNGStreams":
        """Reconstruct RNGStreams from a serialized state produced by
        `to_state`.
        """
        py_rng = None
        np_rng = None
        torch_gen = None
        backend_seed = state.get("backend_seed")

        py_state_b64 = state.get("python_state")
        if py_state_b64 is not None:
            try:
                py_state = pickle.loads(base64.b64decode(py_state_b64.encode("ascii")))
                py_rng = random.Random()
                py_rng.setstate(py_state)
            except Exception:
                py_rng = random.Random()

        np_state_b64 = state.get("numpy_state")
        if np_state_b64 is not None:
            try:
                np_state = pickle.loads(base64.b64decode(np_state_b64.encode("ascii")))
                # Create a new Generator and set its bit_generator state
                np_rng = np.random.default_rng()
                try:
                    np_rng.bit_generator.state = np_state
                except Exception:
                    # fallback: re-seed from backend_seed if available
                    if backend_seed is not None:
                        np_rng = np.random.default_rng(int(backend_seed))
            except Exception:
                np_rng = np.random.default_rng(
                    int(backend_seed) if backend_seed is not None else None
                )
        else:
            np_rng = np.random.default_rng(
                int(backend_seed) if backend_seed is not None else None
            )

        torch_seed = state.get("torch_seed")
        if torch_seed is not None and torch is not None:
            try:
                tg = torch.Generator()
                tg.manual_seed(int(torch_seed))
                torch_gen = tg
            except Exception:
                torch_gen = None

        seeds = state.get("_seeds")
        return RNGStreams(
            python_random_state=py_rng,
            numpy_rng=np_rng,
            torch_generator=torch_gen,
            backend_seed=backend_seed,
            _seeds=seeds,
        )


class RNGManager:
    """Utility that derives and manages per-component RNG streams from a
    single integer seed.

    Implementations should provide deterministic and documented splitting.
    Use this manager to generate seeds for object placement, sensors, and
    backend-specific RNGs (PyBullet, etc.).
    """

    def __init__(self, seed: int):
        self.seed = int(seed)

    def split(self, name: str) -> RNGStreams:
        """Deterministically derive independent RNG streams for a named scope.

        Example: `rng.split('generator')` and `rng.split('sensors')` should
        produce different RNG states but the same results given the same seed
        and name.

        Returns an RNGStreams with objects appropriate for the implementation
        (e.g. numpy.RandomState / numpy.Generator, torch.Generator).
        """

        # Create independent scoped seeds for each component using SHA256.
        def _scoped_int(component: str) -> int:
            h = hashlib.sha256(
                f"{self.seed}:{name}:{component}".encode("utf-8")
            ).hexdigest()
            # Use first 16 hex chars -> 64 bits
            return int(h[:16], 16)

        py_seed = _scoped_int("python")
        np_seed = _scoped_int("numpy")
        torch_seed = _scoped_int("torch")
        backend_seed = _scoped_int("backend")

        # python.random.Random instance
        py_rng = random.Random(py_seed)

        # numpy Generator
        np_rng = np.random.default_rng(np_seed)

        # torch.Generator if available
        torch_gen = None
        if torch is not None:
            try:
                gen = torch.Generator()
                # torch.Generator.manual_seed accepts a single integer (python int)
                gen.manual_seed(int(torch_seed & ((1 << 63) - 1)))
                torch_gen = gen
            except Exception:
                torch_gen = None

        seeds = {
            "python": int(py_seed),
            "numpy": int(np_seed),
            "torch": int(torch_seed),
            "backend": int(backend_seed),
        }
        return RNGStreams(
            python_random_state=py_rng,
            numpy_rng=np_rng,
            torch_generator=torch_gen,
            backend_seed=int(backend_seed),
            _seeds=seeds,
        )


class PhysicsAdapter:
    """Abstract interface for a physics backend adapter.

    Adapters should be responsible for deterministic, fixed-step stepping of
    the underlying physics engine and for exposing spawn/load/save operations.
    """

    def init_world(self, config: Dict, rng: RNGStreams) -> None:
        raise NotImplementedError

    def spawn_object(self, proto: Dict, rng: RNGStreams) -> int:
        """Spawn an object described by `proto` and return an object id."""
        raise NotImplementedError

    def step(self, dt: float) -> None:
        raise NotImplementedError

    def get_state(self) -> Dict:
        """Return a serializable representation of the physics state."""
        raise NotImplementedError

    def set_state(self, state: Dict) -> None:
        raise NotImplementedError

    def apply_action(self, body_index: int, action: Any, mode: str = "torque") -> None:
        """Apply an action to a body in the physics world.

        `body_index` may be an index or a backend-specific id; adapters should
        document semantics. `action` is backend-specific (e.g. list of joint
        torques or positions). `mode` describes how to interpret the action
        (e.g. 'torque'|'position'). This default base method is optional for
        adapters to implement.
        """
        raise NotImplementedError


class Sensor:
    """Base sensor interface. Sensors produce observations (NumPy arrays
    or serializable structures) given a physics world and camera configuration.
    """

    def read(self) -> Observation:
        raise NotImplementedError


def make_env(config: Dict, seed: Optional[int] = None) -> BaseEnv:
    """Factory helper: construct an environment from a config mapping.

    The config should specify generator, physics adapter, sensors and other
    options. The factory returns a ready-to-use BaseEnv instance.
    """
    raise NotImplementedError


class GymWrapper:
    """Lightweight adapter that exposes a BaseEnv as a Gymnasium-compatible
    environment. This module should remain optional (imported only when the
    environment has gymnasium installed) to avoid hard runtime deps.

    Example usage:
        env = make_env(cfg)
        gym_env = GymWrapper(env)
    """

    def __init__(self, env: BaseEnv):
        self._env = env

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Action):
        return self._env.step(action)

    def render(self, mode: str = "rgb_array"):
        return self._env.render(mode=mode)

    def close(self):
        return self._env.close()


class VectorEnv:
    """Minimal vectorized environment wrapper for batching multiple BaseEnv
    instances. The wrapper should present a tensor-first API for efficient
    integration with PyTorch training code.

    This is intentionally lightweight; later we can add multiprocessing,
    async stepping, and shared memory optimizations.
    """

    def __init__(self, envs: Sequence[BaseEnv]):
        self.envs = list(envs)

    def reset(
        self, seeds: Optional[Sequence[Optional[int]]] = None
    ) -> Tuple[Observation, Info]:
        """Reset all envs; seeds can be a sequence with one entry per env."""
        obs = []
        infos = []
        for i, env in enumerate(self.envs):
            seed = None
            if seeds is not None:
                seed = seeds[i]
            o, info = env.reset(seed=seed)
            obs.append(o)
            infos.append(info)
        return obs, {"infos": infos}

    def step(
        self, actions: Sequence[Action]
    ) -> Tuple[Sequence[Observation], Sequence[float], Sequence[bool], Sequence[Info]]:
        """Step each env with the corresponding action."""
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs, rews, dones, infos = zip(*results)
        return list(obs), list(rews), list(dones), list(infos)
