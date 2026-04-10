"""Performance utilities for benchmarking and profiling.

Provides helpers for timing, profiling hotspots, and async I/O for exporters.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional
import functools


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

    def __str__(self):
        return f"{self.name}: {self.elapsed * 1000:.2f}ms"


def timing_decorator(func: Callable) -> Callable:
    """Decorator to time function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed * 1000:.2f}ms")
        return result

    return wrapper


class AsyncExporter:
    """Async wrapper for exporters to avoid blocking simulation.

    Usage:
        async_exporter = AsyncExporter(exporter)
        await async_exporter.write_episode(frames, actions, rewards, dones, metadata)
        await async_exporter.flush()
    """

    def __init__(self, exporter: Any, queue_size: int = 10):
        self.exporter = exporter
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._running = False

    async def write_episode(
        self,
        frames: List[Any],
        actions: List[Any],
        rewards: List[float],
        dones: List[bool],
        metadata: Dict[str, Any],
    ) -> None:
        await self.queue.put((frames, actions, rewards, dones, metadata))

    async def flush(self) -> None:
        while not self.queue.empty():
            item = await self.queue.get()
            frames, actions, rewards, dones, metadata = item
            self.exporter.add_episode(
                frames=frames,
                actions=actions,
                rewards=rewards,
                dones=dones,
                metadata=metadata,
            )


def profile_step_times(
    env_fn: Callable[[], Any],
    num_steps: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Profile environment step times.

    Returns dict with median, p95, p99 step times in ms.
    """
    env = env_fn()
    env.reset()

    # Warmup
    for _ in range(warmup):
        env.step({})

    times = []
    for _ in range(num_steps):
        start = time.perf_counter()
        env.step({})
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    env.close()

    times_ms = [t * 1000 for t in times]
    times_ms.sort()

    return {
        "median": times_ms[len(times_ms) // 2],
        "p95": times_ms[int(len(times_ms) * 0.95)],
        "p99": times_ms[int(len(times_ms) * 0.99)]
        if len(times_ms) > 1
        else times_ms[0],
        "mean": sum(times_ms) / len(times_ms),
    }
