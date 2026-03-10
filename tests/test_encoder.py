import torch
from world_models.vision.dreamer_encoder import ConvEncoder
from world_models.vision.planet_encoder import CNNEncoder


class TestConvEncoder:
    def test_initialization(self):
        input_shape = (3, 64, 64)
        embed_size = 256
        activation = "relu"
        encoder = ConvEncoder(input_shape, embed_size, activation)
        assert encoder.input_shape == input_shape
        assert encoder.embed_size == embed_size
        assert isinstance(encoder.conv_block, torch.nn.Sequential)
        assert isinstance(encoder.fc, torch.nn.Linear)

    def test_forward_pass(self):
        input_shape = (3, 64, 64)
        embed_size = 256
        activation = "relu"
        encoder = ConvEncoder(input_shape, embed_size, activation)
        batch_size = 2
        inputs = torch.randn(batch_size, *input_shape)
        embed = encoder(inputs)
        assert embed.shape == (batch_size, embed_size)

    def test_forward_pass_different_batch(self):
        input_shape = (3, 64, 64)
        embed_size = 256
        activation = "relu"
        encoder = ConvEncoder(input_shape, embed_size, activation)
        inputs = torch.randn(1, *input_shape)
        embed = encoder(inputs)
        assert embed.shape == (1, embed_size)


class TestCNNEncoder:
    def test_initialization(self):
        embedding_size = 256
        activation_function = "relu"
        encoder = CNNEncoder(embedding_size, activation_function)
        assert encoder.embedding_size == embedding_size
        assert isinstance(encoder.conv1, torch.nn.Conv2d)
        assert isinstance(encoder.fc, torch.nn.Linear)

    def test_forward_pass(self):
        embedding_size = 256
        activation_function = "relu"
        encoder = CNNEncoder(embedding_size, activation_function)
        batch_size = 2
        observation = torch.randn(batch_size, 3, 64, 64)
        hidden = encoder(observation)
        assert hidden.shape == (batch_size, embedding_size)

    def test_forward_pass_different_batch(self):
        embedding_size = 256
        activation_function = "relu"
        encoder = CNNEncoder(embedding_size, activation_function)
        observation = torch.randn(1, 3, 64, 64)
        hidden = encoder(observation)
        assert hidden.shape == (1, embedding_size)
