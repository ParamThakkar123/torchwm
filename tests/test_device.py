import pytest
import torch
from unittest.mock import patch, MagicMock


class TestDeviceDetection:
    def test_get_device_returns_cpu_when_no_gpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                from world_models.device import get_device

                device = get_device()
                assert device.type == "cpu"
                assert str(device) == "cpu"

    def test_get_device_returns_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.set_device") as mock_set_device:
                with patch("torch.cuda.device_count", return_value=1):
                    from world_models.device import get_device

                    device = get_device()
                    assert device.type == "cuda"

    def test_get_device_returns_mps_when_cuda_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                from world_models.device import get_device

                device = get_device()
                assert device.type == "mps"

    def test_get_device_respects_preferred_device(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch("torch.cuda.set_device"):
                    from world_models.device import get_device

                    device = get_device("cuda:1")
                    assert device.type == "cuda"
                    assert device.index == 1

    def test_get_device_prefers_cuda_over_mps(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.backends.mps.is_available", return_value=True):
                from world_models.device import get_device

                device = get_device()
                assert device.type == "cuda"


class TestDeviceInfo:
    def test_get_device_info_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                from world_models.device import get_device_info

                info = get_device_info()

                assert info["device_type"] == "cpu"
                assert info["device_name"] == "CPU"
                assert info["is_available"] is False
                assert info["device"] is not None

    def test_get_device_info_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch("torch.cuda.current_device", return_value=0):
                    with patch("torch.cuda.get_device_name", return_value="Test GPU"):
                        with patch("torch.cuda.get_device_properties") as mock_props:
                            mock_props.return_value = MagicMock(
                                total_memory=8 * 1024**3
                            )
                            with patch(
                                "torch.cuda.mem_get_info",
                                return_value=(4 * 1024**3, 8 * 1024**3),
                            ):
                                with patch(
                                    "torch.cuda.get_device_capability",
                                    return_value=(8, 6),
                                ):
                                    from world_models.device import get_device_info

                                    info = get_device_info()

                                    assert info["device_type"] == "cuda"
                                    assert info["device_name"] == "Test GPU"
                                    assert info["is_available"] is True
                                    assert info["device_count"] == 2
                                    assert info["cuda_capability"] == (8, 6)
                                    assert info["cuda_compute_capability"] == "8.6"


class TestSetDeviceForModel:
    def test_set_device_for_model_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                from world_models.device import set_device_for_model

                model = torch.nn.Linear(10, 5)
                result = set_device_for_model(model)

                assert next(model.parameters()).device.type == "cpu"
                assert result is model


class TestSupportsBfloat16:
    def test_supports_bfloat16_cuda_ampere_and_above(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
                with patch("torch.cuda.current_device", return_value=0):
                    from world_models.device import supports_bfloat16

                    assert supports_bfloat16() is True

    def test_supports_bfloat16_cuda_pre_ampere(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
                with patch("torch.cuda.current_device", return_value=0):
                    from world_models.device import supports_bfloat16

                    assert supports_bfloat16() is False

    def test_supports_bfloat16_mps(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                from world_models.device import supports_bfloat16

                assert supports_bfloat16() is True

    def test_supports_bfloat16_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                from world_models.device import supports_bfloat16

                assert supports_bfloat16() is False


class TestSupportsMixedPrecision:
    def test_supports_mixed_precision_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.current_device", return_value=0):
                from world_models.device import supports_mixed_precision

                assert supports_mixed_precision() is True

    def test_supports_mixed_precision_mps(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                from world_models.device import supports_mixed_precision

                assert supports_mixed_precision() is True

    def test_supports_mixed_precision_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                from world_models.device import supports_mixed_precision

                assert supports_mixed_precision() is False


class TestOptimalDeviceIndex:
    def test_get_optimal_device_index_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.current_device", return_value=1):
                from world_models.device import get_optimal_device_index

                assert get_optimal_device_index() == 1

    def test_get_optimal_device_index_mps(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                from world_models.device import get_optimal_device_index

                assert get_optimal_device_index() == 0

    def test_get_optimal_device_index_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                from world_models.device import get_optimal_device_index

                assert get_optimal_device_index() == -1


class TestModuleExports:
    def test_device_functions_exported_from_world_models(self):
        from world_models import get_device, get_device_info, set_device_for_model
        from world_models import supports_bfloat16, supports_mixed_precision

        assert callable(get_device)
        assert callable(get_device_info)
        assert callable(set_device_for_model)
        assert callable(supports_bfloat16)
        assert callable(supports_mixed_precision)
