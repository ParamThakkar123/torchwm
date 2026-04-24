import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from world_models.vision.vq_layer import VectorQuantizer


class LatentActionModel(nn.Module):
    """Latent Action Model (LAM) for unsupervised action learning.

    Learns discrete latent actions from unlabeled video frames using a VQ-VAE
    based objective. The model infers latent actions between frames that encode
    the most meaningful changes for future frame prediction.

    Based on Genie paper - learns actions without action labels from Internet videos.
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 64,
        in_channels: int = 3,
        encoder_dim: int = 256,
        encoder_depth: int = 4,
        num_heads: int = 8,
        patch_size: int = 16,
        vocab_size: int = 8,
        embedding_dim: int = 32,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim

        self.patch_embed = nn.Conv2d(
            in_channels,
            encoder_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (image_size // patch_size) ** 2

        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_dim))
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        self.to_vq_embedding = nn.Linear(encoder_dim, embedding_dim)

        self.vq = VectorQuantizer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            commitment_weight=commitment_weight,
        )

    def encode(
        self, x_prev: torch.Tensor, x_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode frames to latent actions.

        Args:
            x_prev: Previous frames (B, C, T, H, W)
            x_next: Next frame (B, C, H, W)

        Returns:
            latent_actions: Discrete latent action indices (B, T)
            z_q: Quantized embeddings (B, T, H', W', embedding_dim)
        """
        B, C, T, H, W = x_prev.shape

        x_all = torch.cat([x_prev, x_next.unsqueeze(2)], dim=2)

        x = x_all.permute(0, 2, 1, 3, 4).reshape(B * (T + 1), C, H, W)

        x = self.patch_embed(x)
        _, C_enc, H_enc, W_enc = x.shape

        x = x.reshape(B, (T + 1), H_enc * W_enc, C_enc)

        x = x + self.encoder_pos_embed[:, : H_enc * W_enc, :].unsqueeze(1)

        x = x.reshape(B * (T + 1), H_enc * W_enc, C_enc)

        x = self.encoder(x)

        x = x.reshape(B, T + 1, H_enc * W_enc, C_enc)

        x = x[:, :T, :, :]

        z_all = []
        indices_all = []
        for t in range(T):
            x_t = x[:, t, :, :]

            x_t_embed = self.to_vq_embedding(x_t)

            x_t_pooled = x_t_embed.mean(dim=1)

            z_q_t, indices_t, _ = self.vq(x_t_pooled)
            z_all.append(z_q_t)
            indices_all.append(indices_t)

        z_q = torch.stack(z_all, dim=1)
        indices = torch.stack(indices_all, dim=1)

        latent_actions = indices.squeeze(-1)

        return latent_actions, z_q


def create_latent_action_model(
    num_frames: int = 16,
    image_size: int = 64,
    in_channels: int = 3,
    encoder_dim: int = 256,
    encoder_depth: int = 4,
    num_heads: int = 8,
    patch_size: int = 16,
    vocab_size: int = 8,
    embedding_dim: int = 32,
) -> LatentActionModel:
    """Factory function to create a Latent Action Model."""
    return LatentActionModel(
        num_frames=num_frames,
        image_size=image_size,
        in_channels=in_channels,
        encoder_dim=encoder_dim,
        encoder_depth=encoder_depth,
        num_heads=num_heads,
        patch_size=patch_size,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )
