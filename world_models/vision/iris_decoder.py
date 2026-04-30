import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from world_models.vision.iris_encoder import IRISEncoder


class IRISDecoder(nn.Module):
    """CNN Decoder for IRIS discrete autoencoder.

    Decodes discrete tokens back into image observations.
    Uses transposed convolutions to upsample from 4x4 to 64x64.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embedding_dim: int = 512,
        base_channels: int = 32,
        out_channels: int = 3,
        frame_shape: Tuple[int, int, int] = (3, 64, 64),
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.frame_shape = frame_shape
        self.out_channels = out_channels

        # Input projection
        self.input_proj = nn.Conv2d(embedding_dim, embedding_dim, 1)

        # Residual blocks before upsampling
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(embedding_dim) for _ in range(2)]
        )

        # Upsampling blocks (4 -> 8 -> 16 -> 32 -> 64)
        self.upsample_blocks = nn.ModuleList()

        # Block 1: 4x4 -> 8x8
        self.upsample_blocks.append(
            UpsampleBlock(embedding_dim, base_channels * 8, base_channels * 4)
        )

        # Block 2: 8x8 -> 16x16
        self.upsample_blocks.append(
            UpsampleBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        )

        # Block 3: 16x16 -> 32x32
        self.upsample_blocks.append(
            UpsampleBlock(base_channels * 2, base_channels * 2, base_channels)
        )

        # Block 4: 32x32 -> 64x64
        self.upsample_blocks.append(
            UpsampleBlock(base_channels, base_channels, base_channels)
        )

        # Final output projection
        self.output_proj = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode tokens to images.

        Args:
            z: Token embeddings (B, C, H, W) - e.g., (B, 512, 4, 4)

        Returns:
            reconstructed: Reconstructed images (B, C, H, W) - e.g., (B, 3, 64, 64)
        """
        # Project input
        h = self.input_proj(z)

        # Residual blocks
        h = self.residual_blocks(h)

        # Upsampling
        for upsample_block in self.upsample_blocks:
            h = upsample_block(h)

        # Final output
        h = self.output_proj(h)

        # Ensure output matches frame shape
        _, _, out_h, out_w = h.shape
        target_h, target_w = self.frame_shape[1], self.frame_shape[2]

        if out_h != target_h or out_w != target_w:
            h = F.interpolate(
                h, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        return h

    def decode_from_embeddings(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Decode flattened token embeddings to images.

        Args:
            z_flat: Flattened tokens (B, H*W, C) or (B, C, H, W)

        Returns:
            Reconstructed images
        """
        if z_flat.dim() == 3:  # (B, H*W, C)
            B, HW, C = z_flat.shape
            H = W = int(HW**0.5)
            z = z_flat.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            z = z_flat

        return self.forward(z)


class UpsampleBlock(nn.Module):
    """Upsampling block with optional residual connection."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
        )

        # Skip connection projection if needed
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)

        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.skip(x) + self.block(x)


class ResidualBlock(nn.Module):
    """Residual block for decoder."""

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


class DiscreteAutoencoder(nn.Module):
    """Complete Discrete Autoencoder combining encoder and decoder.

    Used for training the VQVAE component of IRIS.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        tokens_per_frame: int = 16,
        embedding_dim: int = 512,
        base_channels: int = 64,
        frame_shape: Tuple[int, int, int] = (3, 64, 64),
    ):
        super().__init__()

        self.encoder = IRISEncoder(
            vocab_size=vocab_size,
            tokens_per_frame=tokens_per_frame,
            embedding_dim=embedding_dim,
            in_channels=frame_shape[0],
            base_channels=base_channels,
            frame_shape=frame_shape,
        )

        self.decoder = IRISDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            base_channels=32,  # decoder uses smaller channels
            out_channels=frame_shape[0],
            frame_shape=frame_shape,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Full encode-decode forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            reconstruction: Reconstructed images
            indices: Token indices (B, H', W')
            loss_dict: Dictionary with loss components
        """
        z_q, indices, vq_loss = self.encoder(x)

        # Decode (use detached z_q to stop gradient through decoder for VQ loss)
        self.decoder(
            z_q.detach() + z_q - z_q
        )  # identity with gradient stop for z_q part

        # Actually, we want gradients to flow through reconstruction path
        reconstruction_st = self.decoder(z_q)

        # Compute reconstruction loss
        recon_loss = F.l1_loss(reconstruction_st, x)

        # Combine losses
        loss = recon_loss + vq_loss["vq_loss"]

        loss_dict = {
            "reconstruction": recon_loss,
            "vq": vq_loss["vq_loss"],
            "perplexity": vq_loss["perplexity"],
            "total": loss,
        }

        return reconstruction_st, indices, loss_dict

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to token indices."""
        return self.encoder.encode_to_indices(x)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to images."""
        embeddings = self.encoder.decode_from_indices(indices)
        return self.decoder(embeddings)
