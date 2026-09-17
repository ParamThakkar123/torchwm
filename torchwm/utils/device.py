"""Device selection shared by models, trainers and the CLI."""

from __future__ import annotations

import torch


def mps_is_available() -> bool:
    """Whether Apple's Metal (MPS) backend can be used."""
    backend = getattr(torch.backends, "mps", None)
    try:
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def get_default_device(allow_gpu: bool = True) -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if allow_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
    return torch.device("cpu")


def default_device_name(allow_gpu: bool = True) -> str:
    """String form of :func:`get_default_device`, for configs and CLI defaults."""
    return get_default_device(allow_gpu).type


def resolve_device(requested: str | torch.device | None) -> torch.device:
    """Honour a requested device, falling back when it is not available.

    ``None`` selects :func:`get_default_device`. A CUDA request on a machine
    without CUDA, or an MPS request without MPS, falls back to the best device
    that does exist instead of failing later inside ``.to(device)``.
    """
    if requested is None:
        return get_default_device()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return get_default_device()
    if device.type == "mps" and not mps_is_available():
        return torch.device("cpu")
    return device


__all__ = [
    "default_device_name",
    "get_default_device",
    "mps_is_available",
    "resolve_device",
]
