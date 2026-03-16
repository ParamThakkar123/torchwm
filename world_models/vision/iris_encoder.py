import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from world_models.vision.vq_layer import VectorQuantizerEMA


class IRISEncoder(nn.Module):
    """CNN Encoder for IRIS discrete autoencoder.

    Encodes image observations into latent features, which are then quantized
    into discrete tokens using the VectorQuantizer.

    Architecture:
        - 4 convolutional layers with residual blocks
        - Self-attention at 8x8 and 16x16 resolutions
        - Vector quantization to produce discrete tokens
    """

    def __init__(
        self,
        vocab_size: int = 512,
        tokens_per_frame: int = 16,
        embedding_dim: int = 512,
        in_channels: int = 3,
        base_channels: int = 64,
        num_residual_blocks: int = 2,
        frame_shape: Tuple[int, int, int] = (3, 64, 64),
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.tokens_per_frame = tokens_per_frame
        self.embedding_dim = embedding_dim

        # Compute expected spatial dimensions after conv layers
        # 64 -> 32 -> 16 -> 8 -> 4 with 4 conv layers
        self.expected_spatial_size = 4  # After 4 stride-2 convs, 64/16 = 4

        # Number of tokens per dimension (height x width)
        # tokens_per_frame should equal expected_spatial_size^2
        # 16 = 4x4

        # CNN encoder body
        self.conv_blocks = nn.ModuleList()
        in_ch = in_channels

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        for i, out_ch in enumerate(channels):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    nn.ReLU(),
                )
            )
            in_ch = out_ch

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels[-1]) for _ in range(num_residual_blocks)]
        )

        # Self-attention at intermediate resolutions
        # Apply attention at 8x8 and 16x16
        self.attention_8 = SelfAttentionBlock(channels[1])  # 16x16
        self.attention_4 = SelfAttentionBlock(channels[2])  # 8x8

        # Project to embedding dimension
        self.projection = nn.Conv2d(channels[-1], embedding_dim, 1)

        # Vector quantizer
        self.quantizer = VectorQuantizerEMA(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            commitment_weight=0.25,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Encode images to discrete tokens.

        Args:
            x: Input images (B, C, H, W) - should be 64x64

        Returns:
            z_q: Quantized tokens (B, C, H', W')
            indices: Token indices (B, H', W')
            vq_loss: Dictionary with VQ loss components
        """
        # CNN encoding
        h = x

        # Conv block 1 -> 32x32
        h = self.conv_blocks[0](h)

        # Conv block 2 -> 16x16, apply attention at this resolution
        h = self.conv_blocks[1](h)
        h = self.attention_8(h)

        # Conv block 3 -> 8x8, apply attention at this resolution
        h = self.conv_blocks[2](h)
        h = self.attention_4(h)

        # Conv block 4 -> 4x4
        h = self.conv_blocks[3](h)

        # Residual blocks
        h = self.residual_blocks(h)

        # Project to embedding dimension
        h = self.projection(h)

        # Quantize
        z_q, indices, vq_loss = self.quantizer(h)

        return z_q, indices, vq_loss

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Encode directly to token indices (for world model)."""
        with torch.no_grad():
            _, indices, _ = self.forward(x)
        return indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to embeddings (for decoder)."""
        return self.quantizer.decode_indices(indices)


class ResidualBlock(nn.Module):
    """Residual block for encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SelfAttentionBlock(nn.Module):
    """Self-attention block for encoder.

    Applies spatial self-attention to capture long-range dependencies.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Compute Q, K, V
        q = self.query(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = self.key(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        v = self.value(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        # Attention scores
        attn = torch.bmm(q, k.transpose(1, 2)) / (C**0.5)
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = torch.bmm(attn, v)  # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        # Residual connection with learned weight
        return x + self.gamma * out
