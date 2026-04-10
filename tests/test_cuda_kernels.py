import pytest
import torch
import numpy as np
from world_models.cuda_kernels import batched_normalize, batched_add_noise, HAS_CUDA


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestCudaKernels:
    def test_batched_normalize_cuda(self):
        """Test CUDA batched_normalize kernel."""
        batch_size, channels, height, width = 4, 3, 8, 8
        data = torch.randn(batch_size, channels, height, width, device="cuda")

        # Apply kernel
        result = batched_normalize(data)

        # Should be on CUDA
        assert result.is_cuda

        # Should be clamped to [0, 1]
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

        # Shape should be preserved
        assert result.shape == data.shape

    def test_batched_normalize_fallback(self):
        """Test fallback when CUDA not available."""
        if HAS_CUDA:
            pytest.skip("CUDA available, testing CUDA version")

        batch_size, channels, height, width = 4, 3, 8, 8
        data = torch.randn(batch_size, channels, height, width)

        result = batched_normalize(data)

        # Should be clamped to [0, 1]
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

        assert result.shape == data.shape

    def test_batched_add_noise_cuda(self):
        """Test CUDA batched_add_noise kernel."""
        batch_size, channels, height, width = 4, 3, 8, 8
        data = torch.randn(batch_size, channels, height, width, device="cuda")
        noise = torch.randn_like(data)

        result = batched_add_noise(data, noise)

        # Should be on CUDA
        assert result.is_cuda

        # Should equal data + noise
        expected = data + noise
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

        # Shape should be preserved
        assert result.shape == data.shape

    def test_batched_add_noise_fallback(self):
        """Test fallback for batched_add_noise."""
        if HAS_CUDA:
            pytest.skip("CUDA available, testing CUDA version")

        batch_size, channels, height, width = 4, 3, 8, 8
        data = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(data)

        result = batched_add_noise(data, noise)

        expected = data + noise
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

        assert result.shape == data.shape

    def test_normalize_edge_cases(self):
        """Test batched_normalize with edge case values."""
        # Test with values outside [0, 1]
        data = torch.tensor(
            [[[[-2.0, 0.5, 1.5]]]],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        result = batched_normalize(data)

        expected = torch.tensor([[[[0.0, 0.5, 1.0]]]], device=result.device)
        torch.testing.assert_close(result, expected)

    def test_add_noise_shapes(self):
        """Test batched_add_noise with different shapes."""
        shapes = [
            (2, 3, 4, 4),
            (1, 1, 8, 8),
            (8, 3, 16, 16),
        ]

        for shape in shapes:
            data = torch.randn(
                *shape, device="cuda" if torch.cuda.is_available() else "cpu"
            )
            noise = torch.randn_like(data)

            result = batched_add_noise(data, noise)
            expected = data + noise

            torch.testing.assert_close(result, expected)
            assert result.shape == shape
