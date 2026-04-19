import torch
import torchvision.transforms as T
from .base import OperatorABC
from typing import Dict, Any


class DreamerOperator(OperatorABC):
    """Operator for Dreamer model preprocessing: normalizes observations and encodes actions."""

    def __init__(self, image_size: int = 64, action_dim: int = 6):
        self.image_size = image_size
        self.action_dim = action_dim
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def process(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process Dreamer inputs: image observation and action.

        Expected inputs: {'image': PIL.Image or tensor, 'action': tensor or list}
        """
        processed = {}

        # Process image
        if "image" in inputs:
            if isinstance(inputs["image"], torch.Tensor):
                processed["obs"] = (
                    inputs["image"].unsqueeze(0)
                    if inputs["image"].dim() == 3
                    else inputs["image"]
                )
            else:
                # Assume PIL Image
                processed["obs"] = self.transform(inputs["image"]).unsqueeze(0)

        # Process action
        if "action" in inputs:
            action = inputs["action"]
            if isinstance(action, list):
                action = torch.tensor(action, dtype=torch.float32)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            processed["action"] = action

        return processed
