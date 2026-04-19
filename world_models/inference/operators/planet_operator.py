import torch
import torchvision.transforms as T
from .base import OperatorABC
from typing import Dict, Any


class PlaNetOperator(OperatorABC):
    """Operator for PlaNet model preprocessing: encodes environment states and transitions."""

    def __init__(self, state_dim: int = 32, action_dim: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),  # Assuming grayscale for simplicity
            ]
        )

    def process(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process PlaNet inputs: state observations and actions.

        Expected inputs: {'obs': tensor or image, 'action': tensor, 'reward': float, 'done': bool}
        """
        processed = {}

        # Process observation
        if "obs" in inputs:
            obs = inputs["obs"]
            if isinstance(obs, torch.Tensor):
                processed["obs"] = obs.unsqueeze(0) if obs.dim() == 2 else obs
            else:
                # Assume image
                processed["obs"] = self.transform(obs).unsqueeze(0)

        # Process action
        if "action" in inputs:
            action = inputs["action"]
            if isinstance(action, list):
                action = torch.tensor(action, dtype=torch.float32)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            processed["action"] = action

        # Process reward
        if "reward" in inputs:
            reward = torch.tensor([inputs["reward"]], dtype=torch.float32)
            processed["reward"] = reward

        # Process done
        if "done" in inputs:
            done = torch.tensor([inputs["done"]], dtype=torch.float32)
            processed["done"] = done

        return processed
