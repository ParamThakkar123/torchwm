"""TFRecord exporter for TensorFlow pipelines.

Stores episodes as TFRecord sequences (serialized Example protos) for large-scale
TensorFlow input pipelines.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

try:
    import tensorflow as tf  # type: ignore[assignment]
    import numpy as np  # type: ignore[assignment]
except Exception:  # pragma: no cover - optional dependency
    tf = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


class TFRecordExporter:
    """Export episodes to TFRecord format.

    Each episode is serialized as a tf.train.Example with feature tensors
    for frames, actions, rewards, dones.
    """

    def __init__(self, path: str, compression_type: str = "ZLIB"):
        if tf is None or np is None:
            raise ImportError("TensorFlow and numpy are required for TFRecordExporter")
        self.path = path
        self.compression_type = compression_type
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._writer = tf.io.TFRecordWriter(
            path, options=tf.io.TFRecordOptions(compression_type=compression_type)
        )

    def close(self) -> None:
        try:
            self._writer.close()
        except Exception:
            pass

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def add_episode(
        self,
        frames: Sequence[Any],
        actions: Optional[Sequence[Any]] = None,
        rewards: Optional[Sequence[float]] = None,
        dones: Optional[Sequence[bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add an episode to the TFRecord file. Returns episode index."""
        frames_arr = np.asarray(frames)
        if frames_arr.dtype != np.uint8:
            if np.issubdtype(frames_arr.dtype, np.floating):
                frames_arr = (frames_arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frames_arr = frames_arr.astype(np.uint8)

        T, H, W, C = frames_arr.shape

        # Serialize frames as JPEG to save space
        encoded_frames = []
        for t in range(T):
            img = tf.image.encode_jpeg(frames_arr[t])
            encoded_frames.append(img.numpy())

        actions_arr = (
            np.asarray(actions, dtype=np.float32)
            if actions is not None
            else np.zeros((T, 0), dtype=np.float32)
        )
        rewards_arr = (
            np.asarray(rewards, dtype=np.float32)
            if rewards is not None
            else np.zeros(T, dtype=np.float32)
        )
        dones_arr = (
            np.asarray(dones, dtype=np.int32)
            if dones is not None
            else np.zeros(T, dtype=np.int32)
        )

        feature = {
            "frames": self._bytes_feature(np.concatenate(encoded_frames).tobytes()),
            "frames_shape": self._int64_feature([T, H, W, C]),
            "actions": self._bytes_feature(actions_arr.tobytes()),
            "actions_shape": self._int64_feature(list(actions_arr.shape)),
            "rewards": self._float_feature(rewards_arr.tolist()),
            "dones": self._int64_feature([int(d) for d in dones_arr.tolist()]),
        }

        if metadata:
            # Store metadata as JSON string
            import json

            meta_str = json.dumps(metadata)
            feature["metadata"] = self._bytes_feature(meta_str.encode("utf-8"))

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self._writer.write(example.SerializeToString())
        self._writer.flush()

        return 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
