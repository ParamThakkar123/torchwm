"""HDF5 exporter for episodic simulation data.

This module provides a simple HDF5-backed exporter that writes one group
per episode. Each episode group contains:
  - frames: [T, H, W, C] uint8
  - actions: [T, ...] (float32)
  - rewards: [T] (float32)
  - dones: [T] (uint8)
  - metadata: JSON string attribute on the episode group

The exporter uses gzip compression for frames by default.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence
import os

try:
    import h5py
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    h5py = None
    np = None


class HDF5Exporter:
    def __init__(self, path: str, mode: str = "a"):
        if h5py is None:
            raise ImportError("h5py and numpy are required for HDF5Exporter")
        self.path = path
        self.mode = mode
        # ensure parent dir exists
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = h5py.File(path, mode)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def _next_episode_index(self) -> int:
        # find next available numeric episode id
        existing = [
            int(k.replace("episode_", ""))
            for k in self._f.keys()
            if k.startswith("episode_")
        ]
        return max(existing) + 1 if existing else 0

    def add_episode(
        self,
        frames: Sequence[Any],
        actions: Optional[Sequence[Any]] = None,
        rewards: Optional[Sequence[float]] = None,
        dones: Optional[Sequence[bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add an episode to the HDF5 file. Returns episode index."""
        idx = self._next_episode_index()
        grp = self._f.create_group(f"episode_{idx}")

        frames_arr = np.asarray(frames)
        # store frames as uint8 if possible
        if frames_arr.dtype != np.uint8:
            if np.issubdtype(frames_arr.dtype, np.floating):
                frames_arr = np.clip(frames_arr * 255.0, 0, 255).astype(np.uint8)
            else:
                frames_arr = frames_arr.astype(np.uint8)

        grp.create_dataset(
            "frames", data=frames_arr, compression="gzip", compression_opts=4
        )

        T = frames_arr.shape[0]

        if actions is None:
            actions_arr = np.zeros((T, 0), dtype=np.float32)
        else:
            actions_arr = np.asarray(actions, dtype=np.float32)
        grp.create_dataset("actions", data=actions_arr, compression="gzip")

        if rewards is None:
            rewards_arr = np.zeros((T,), dtype=np.float32)
        else:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
        grp.create_dataset("rewards", data=rewards_arr, compression="gzip")

        if dones is None:
            dones_arr = np.zeros((T,), dtype=np.uint8)
        else:
            dones_arr = np.asarray(dones, dtype=np.uint8)
        grp.create_dataset("dones", data=dones_arr, compression="gzip")

        meta = metadata or {}
        grp.attrs["metadata"] = json.dumps(meta)

        # flush to disk
        self._f.flush()
        return idx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
