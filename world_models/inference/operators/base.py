from abc import ABC, abstractmethod
import torch
from typing import Any, Dict, Union


class OperatorABC(ABC):
    """Abstract base class for operators that preprocess inputs for inference pipelines."""

    @abstractmethod
    def process(self, inputs: Any) -> Dict[str, torch.Tensor]:
        """
        Process raw inputs into standardized tensor format for model consumption.

        Args:
            inputs: Raw input data (dict, tensor, or other formats)

        Returns:
            Dict of processed tensors ready for model input
        """
        pass

    def __call__(self, inputs: Any) -> Dict[str, torch.Tensor]:
        return self.process(inputs)
