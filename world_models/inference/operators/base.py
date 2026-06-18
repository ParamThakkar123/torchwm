from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TensorSpec:
    """Optional tensor contract used to validate operator inputs or outputs.

    Args:
        shape: Expected shape. Use ``None`` as a wildcard for dimensions that may
            vary, such as batch size.
        dtype: Expected tensor dtype.
        required: Whether the key must be present in the mapping being validated.
    """

    shape: tuple[int | None, ...] | None = None
    dtype: torch.dtype | None = None
    required: bool = True


class OperatorABC(nn.Module, ABC):
    """Structured base class for inference operators.

    Operators use a consistent pipeline:

    1. ``preprocess`` converts raw inputs into tensors.
    2. ``forward`` performs model/operator-specific tensor computation.
    3. ``postprocess`` formats the final output mapping.

    Subclasses may also declare ``input_specs`` and ``output_specs`` to validate
    required tensor keys, shapes, and dtypes. ``OperatorABC`` inherits from
    ``torch.nn.Module``, so operators support ``to(device)``, ``train()``, and
    ``eval()`` just like model modules.
    """

    input_specs: Mapping[str, TensorSpec] = {}
    output_specs: Mapping[str, TensorSpec] = {}

    def __init__(self, *, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

    @abstractmethod
    def preprocess(self, inputs: Any) -> dict[str, torch.Tensor]:
        """Convert raw inputs into a tensor mapping ready for ``forward``."""

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run tensor computation for this operator.

        Preprocessing-only operators can rely on this identity implementation.
        Operators that wrap a model should override this method.
        """

        return inputs

    def postprocess(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Format validated forward outputs for consumers."""

        return outputs

    def process(self, inputs: Any) -> dict[str, torch.Tensor]:
        """Process raw inputs through preprocess, forward, and postprocess stages."""

        preprocessed = self.preprocess(inputs)
        self.validate_mapping(
            preprocessed, self.input_specs, label="preprocessed input"
        )
        preprocessed = self._move_tensors(preprocessed)
        outputs = self.forward(preprocessed)
        self.validate_mapping(outputs, self.output_specs, label="operator output")
        return self.postprocess(outputs)

    def batch(self, inputs: Sequence[Any]) -> dict[str, torch.Tensor]:
        """Preprocess a sequence of inputs and stack matching tensor keys."""

        if not inputs:
            raise ValueError("Cannot batch an empty input sequence")
        processed = [self.process(item) for item in inputs]
        keys = processed[0].keys()
        for index, item in enumerate(processed[1:], start=1):
            if item.keys() != keys:
                raise ValueError(
                    f"Batched operator outputs must share keys; item 0 has {sorted(keys)} "
                    f"but item {index} has {sorted(item.keys())}"
                )
        return {key: torch.stack([item[key] for item in processed]) for key in keys}

    def to(self, *args: Any, **kwargs: Any) -> "OperatorABC":
        """Move module parameters/buffers and remember the target tensor device."""

        module = super().to(*args, **kwargs)
        device = self._device_from_to_args(*args, **kwargs)
        if device is not None:
            self.device = device
        return module

    def __call__(self, inputs: Any) -> dict[str, torch.Tensor]:
        return self.process(inputs)

    @classmethod
    def validate_mapping(
        cls,
        values: Mapping[str, torch.Tensor],
        specs: Mapping[str, TensorSpec],
        *,
        label: str,
    ) -> None:
        """Validate tensor keys, shapes, and dtypes against optional specs."""

        if not isinstance(values, Mapping):
            raise TypeError(f"{label} must be a mapping of tensor names to tensors")
        for key, spec in specs.items():
            if key not in values:
                if spec.required:
                    raise ValueError(f"Missing required {label} key: {key!r}")
                continue
            value = values[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{label} key {key!r} must be a torch.Tensor")
            if spec.dtype is not None and value.dtype != spec.dtype:
                raise TypeError(
                    f"{label} key {key!r} must have dtype {spec.dtype}, got {value.dtype}"
                )
            if spec.shape is not None:
                cls._validate_shape(key, value, spec.shape, label=label)

    @staticmethod
    def _validate_shape(
        key: str,
        value: torch.Tensor,
        expected: tuple[int | None, ...],
        *,
        label: str,
    ) -> None:
        if value.dim() != len(expected):
            raise ValueError(
                f"{label} key {key!r} must have {len(expected)} dims, got {value.dim()}"
            )
        for dim_index, (actual, expected_dim) in enumerate(zip(value.shape, expected)):
            if expected_dim is not None and actual != expected_dim:
                raise ValueError(
                    f"{label} key {key!r} dim {dim_index} must be {expected_dim}, got {actual}"
                )

    def _move_tensors(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in values.items()}

    @staticmethod
    def _device_from_to_args(*args: Any, **kwargs: Any) -> torch.device | None:
        if "device" in kwargs and kwargs["device"] is not None:
            return torch.device(kwargs["device"])
        for arg in args:
            if isinstance(arg, (torch.device, str, int)):
                try:
                    return torch.device(arg)
                except (TypeError, RuntimeError):
                    continue
            if isinstance(arg, torch.Tensor):
                return arg.device
        return None
