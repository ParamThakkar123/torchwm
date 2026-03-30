"""Image + JSON exporter.

Saves per-episode frames as PNG images and a sidecar JSON file containing
actions, rewards, dones and metadata. Useful for quick inspection or when
building datasets incrementally.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

try:
    from PIL import Image
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    Image = None
    np = None


class ImageJSONExporter:
    def __init__(self, out_dir: str, fmt: str = "png") -> None:
        if Image is None or np is None:
            raise ImportError("Pillow and numpy are required for ImageJSONExporter")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.fmt = fmt.lower().lstrip(".")

    def _next_episode_index(self) -> int:
        existing = [d for d in os.listdir(self.out_dir) if d.startswith("episode_")]
        ids = []
        for d in existing:
            try:
                ids.append(int(d.replace("episode_", "")))
            except Exception:
                pass
        return max(ids) + 1 if ids else 0

    def add_episode(
        self,
        frames: Sequence[Any],
        actions: Optional[Sequence[Any]] = None,
        rewards: Optional[Sequence[float]] = None,
        dones: Optional[Sequence[bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        idx = self._next_episode_index()
        ep_dir = os.path.join(self.out_dir, f"episode_{idx}")
        os.makedirs(ep_dir, exist_ok=True)

        frames_arr = np.asarray(frames)
        # Ensure uint8
        if frames_arr.dtype != np.uint8:
            if np.issubdtype(frames_arr.dtype, np.floating):
                frames_arr = (frames_arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frames_arr = frames_arr.astype(np.uint8)

        T = frames_arr.shape[0]
        for t in range(T):
            img = Image.fromarray(frames_arr[t])
            fname = os.path.join(ep_dir, f"frame_{t:06d}.{self.fmt}")
            img.save(fname)

        meta = metadata or {}
        meta.update({"frames": T})
        meta_record = {
            "metadata": meta,
            "actions": actions or [],
            "rewards": list(rewards or []),
            "dones": list(dones or []),
        }

        with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
            json.dump(meta_record, f, indent=2)

        return idx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # nothing to cleanup
        return False
