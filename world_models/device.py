import torch
from typing import Optional


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """Get the best available device for training.

    Automatically detects and prioritizes available GPUs in order:
    1. NVIDIA GPUs (CUDA)
    2. AMD GPUs (ROCm/HIP)
    3. Apple Silicon GPUs (MPS)
    4. CPU (fallback)

    Args:
        preferred_device: Optional device string preference (e.g., "cuda:0", "mps").
                        If specified and available, will use this device.

    Returns:
        torch.device: The best available device
    """
    if preferred_device is not None:
        try:
            requested = torch.device(preferred_device)
            if requested.type == "cuda":
                if torch.cuda.is_available():
                    cuda_idx = requested.index or 0
                    if cuda_idx < torch.cuda.device_count():
                        torch.cuda.set_device(requested)
                        return requested
            elif requested.type == "mps":
                if torch.backends.mps.is_available():
                    return requested
        except (RuntimeError, AssertionError):
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device_info() -> dict:
    """Get information about the available compute device.

    Returns:
        dict: Device information including type, name, memory, and capability
    """
    info = {
        "device": None,
        "device_type": "cpu",
        "device_name": "CPU",
        "device_count": 1,
        "current_device": 0,
        "memory_total": 0,
        "memory_available": 0,
        "cuda_capability": None,
        "is_available": False,
    }

    if torch.cuda.is_available():
        info["device"] = torch.device("cuda")
        info["device_type"] = "cuda"
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(info["current_device"])
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
        info["memory_available"] = torch.cuda.mem_get_info()[0]
        info["cuda_capability"] = torch.cuda.get_device_capability()
        info["is_available"] = True

        major, minor = info["cuda_capability"]
        info["cuda_compute_capability"] = f"{major}.{minor}"

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = torch.device("mps")
        info["device_type"] = "mps"
        info["device_name"] = "Apple Silicon GPU"
        info["is_available"] = True

    else:
        info["device"] = torch.device("cpu")
        info["device_type"] = "cpu"
        info["device_name"] = "CPU"

    return info


def set_device_for_model(
    model: torch.nn.Module, device: Optional[torch.device] = None
) -> torch.nn.Module:
    """Move a model to the specified device and return it.

    Args:
        model: PyTorch model to move
        device: Target device. If None, auto-detects best device.

    Returns:
        The model moved to the device
    """
    if device is None:
        device = get_device()

    return model.to(device)


def get_optimal_device_index() -> int:
    """Get the index of the optimal device for single-GPU training.

    Returns:
        int: Device index (0 for CUDA, 0 for MPS, -1 for CPU)
    """
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 0
    return -1


def supports_bfloat16(device: Optional[torch.device] = None) -> bool:
    """Check if the device supports bfloat16 precision.

    Args:
        device: Device to check. If None, uses best available device.

    Returns:
        bool: True if bfloat16 is supported
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        return capability[0] >= 8

    elif device.type == "mps":
        return True

    return False


def supports_mixed_precision(device: Optional[torch.device] = None) -> bool:
    """Check if the device supports mixed precision training.

    Args:
        device: Device to check. If None, uses best available device.

    Returns:
        bool: True if mixed precision is supported
    """
    if device is None:
        device = get_device()

    return device.type in ("cuda", "mps")
