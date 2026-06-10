import pytest
import torch
from world_models.inference.operators import (
    DreamerOperator,
    JEPAOperator,
    IrisOperator,
    PlaNetOperator,
)
from PIL import Image
import numpy as np


class TestOperators:
    def test_dreamer_operator(self):
        operator = DreamerOperator()

        # Test with dummy tensor
        inputs = {
            "image": torch.randn(3, 64, 64),
            "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
        result = operator.process(inputs)
        assert "obs" in result
        assert "action" in result
        assert result["obs"].shape == (1, 3, 64, 64)
        assert result["action"].shape == (1, 6)

    def test_jepa_operator(self):
        operator = JEPAOperator()

        # Test with dummy images
        dummy_img = torch.randn(3, 224, 224)
        inputs = {"images": [dummy_img]}
        result = operator.process(inputs)
        assert "images" in result
        assert "mask" in result
        assert result["images"].shape == (1, 3, 224, 224)

    def test_iris_operator(self):
        operator = IrisOperator()

        # Test with dummy tokens
        inputs = {"tokens": [1, 2, 3, 4, 5]}
        result = operator.process(inputs)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (1, 512)
        assert result["attention_mask"].shape == (1, 512)

    def test_planet_operator(self):
        operator = PlaNetOperator()

        # Test with dummy inputs
        inputs = {
            "obs": torch.randn(32),
            "action": [0.1, 0.2],
            "reward": 1.0,
            "done": False,
        }
        result = operator.process(inputs)
        assert "obs" in result
        assert "action" in result
        assert "reward" in result
        assert "done" in result
