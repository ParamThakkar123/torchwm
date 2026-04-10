"""
Custom CUDA kernels for TorchWM performance optimization.
"""

import torch
import importlib.util
import os

# Check for CUDA
HAS_CUDA = torch.cuda.is_available()

if HAS_CUDA:
    try:
        # Try importing pre-compiled extension
        import torchwm_cuda_kernels as cuda_extension

        batched_normalize_cuda = cuda_extension.batched_normalize_cuda
        batched_add_noise_cuda = cuda_extension.batched_add_noise_cuda
    except ImportError:
        try:
            from torch.utils.cpp_extension import load

            # Get the directory of this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cuda_dir = os.path.join(current_dir, "cuda")

            # Load the extension from cuda/ directory
            cuda_extension = load(
                name="torchwm_cuda_kernels",
                sources=[
                    os.path.join(cuda_dir, "kernels.cu"),
                ],
                verbose=False,
            )

            batched_normalize_cuda = cuda_extension.batched_normalize_cuda
            batched_add_noise_cuda = cuda_extension.batched_add_noise_cuda

        except Exception as e:
            print(f"Warning: Failed to load CUDA kernels: {e}")
            HAS_CUDA = False

else:
    batched_normalize_cuda = None
    batched_add_noise_cuda = None


def batched_normalize(data: torch.Tensor) -> torch.Tensor:
    """Apply batched normalization using CUDA kernel if available."""
    if HAS_CUDA and batched_normalize_cuda is not None and data.is_cuda:
        return batched_normalize_cuda(data)
    else:
        # Fallback to CPU/PyTorch implementation
        return torch.clamp(data, 0.0, 1.0)


def batched_add_noise(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Add noise to data using CUDA kernel if available."""
    if (
        HAS_CUDA
        and batched_add_noise_cuda is not None
        and data.is_cuda
        and noise.is_cuda
    ):
        return batched_add_noise_cuda(data, noise)
    else:
        # Fallback
        return data + noise
