"""
Benchmark script for CUDA kernels vs PyTorch implementations.
"""

import torch
import time
import numpy as np
from world_models.cuda_kernels import batched_normalize, batched_add_noise, HAS_CUDA


def benchmark_normalize(batch_size=32, channels=3, height=64, width=64, num_runs=100):
    """Benchmark batched_normalize kernel vs PyTorch clamp."""
    print(f"Benchmarking batched_normalize: {batch_size}x{channels}x{height}x{width}")

    # Create test data
    data = torch.randn(
        batch_size,
        channels,
        height,
        width,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # PyTorch implementation
    def pytorch_normalize(data):
        return torch.clamp(data, 0.0, 1.0)

    # CUDA kernel implementation (if available)
    def cuda_normalize(data):
        return batched_normalize(data)

    # Warmup
    _ = pytorch_normalize(data)
    if HAS_CUDA:
        _ = cuda_normalize(data)

    # Benchmark PyTorch
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(num_runs):
        result_pytorch = pytorch_normalize(data)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pytorch_time = (time.time() - start) / num_runs

    # Benchmark CUDA
    if HAS_CUDA:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            result_cuda = cuda_normalize(data)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / num_runs

        # Verify results are close
        torch.testing.assert_close(result_pytorch, result_cuda, rtol=1e-5, atol=1e-5)
        speedup = pytorch_time / cuda_time
        print(".4f")
        print(".4f")
        print(".2f")
    else:
        print(".4f")
        print("CUDA kernels not available")


def benchmark_add_noise(batch_size=32, channels=3, height=64, width=64, num_runs=100):
    """Benchmark batched_add_noise kernel vs PyTorch add."""
    print(f"Benchmarking batched_add_noise: {batch_size}x{channels}x{height}x{width}")

    # Create test data
    data = torch.randn(
        batch_size,
        channels,
        height,
        width,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    noise = torch.randn_like(data)

    # PyTorch implementation
    def pytorch_add_noise(data, noise):
        return data + noise

    # CUDA kernel implementation
    def cuda_add_noise(data, noise):
        return batched_add_noise(data, noise)

    # Warmup
    _ = pytorch_add_noise(data, noise)
    if HAS_CUDA:
        _ = cuda_add_noise(data, noise)

    # Benchmark PyTorch
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(num_runs):
        result_pytorch = pytorch_add_noise(data, noise)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pytorch_time = (time.time() - start) / num_runs

    # Benchmark CUDA
    if HAS_CUDA:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            result_cuda = cuda_add_noise(data, noise)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / num_runs

        # Verify results are close
        torch.testing.assert_close(result_pytorch, result_cuda, rtol=1e-5, atol=1e-5)
        speedup = pytorch_time / cuda_time
        print(".4f")
        print(".4f")
        print(".2f")
    else:
        print(".4f")
        print("CUDA kernels not available")


if __name__ == "__main__":
    print("TorchWM CUDA Kernels Benchmark")
    print("=" * 40)

    # Test different sizes
    sizes = [
        (8, 3, 32, 32),
        (16, 3, 64, 64),
        (32, 3, 64, 64),
        (64, 3, 128, 128),
    ]

    for batch_size, channels, height, width in sizes:
        print(f"\n--- Batch size: {batch_size} ---")
        benchmark_normalize(batch_size, channels, height, width)
        benchmark_add_noise(batch_size, channels, height, width)
