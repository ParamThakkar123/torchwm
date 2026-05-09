import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from world_models.vision.vq_layer import VectorQuantizer


class LatentActionModel(nn.Module):
    """Latent Action Model (LAM) for unsupervised action learning.

    Learns discrete latent actions from unlabeled video frames using a VQ-VAE
    based objective. The model infers latent actions between frames that encode
    the most meaningful changes for future frame prediction.

    Based on Genie paper - learns actions without action labels from Internet videos.

    Components:
    - Encoder: Takes all previous frames x1:t and next frame x_t+1 → outputs latent actions
    - Decoder: Takes previous frames x1:t-1 and latent actions a1:t-1 → predicts next frame x_t
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 64,
        in_channels: int = 3,
        encoder_dim: int = 256,
        decoder_dim: int = 512,
        encoder_depth: int = 4,
        decoder_depth: int = 4,
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
        self.decoder_dim = decoder_dim
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        num_patches = (image_size // patch_size) ** 2

        # ===== ENCODER =====
        # Takes x1:t and x_t+1, outputs latent actions at each timestep
        self.patch_embed = nn.Conv2d(
            in_channels,
            encoder_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

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

        # ===== ACTION EMBEDDING =====
        self.action_embedding = nn.Embedding(vocab_size, encoder_dim)

        # ===== DECODER =====
        # Takes x1:t-1 and a1:t-1, predicts x_t
        # Uses masked frames (all but first) to force action usage
        self.decoder_patch_embed = nn.Conv2d(
            in_channels,
            decoder_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Decoder uses action embeddings as additive conditioning
        self.decoder_action_proj = nn.Linear(embedding_dim, decoder_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # Output projection to reconstruct pixels
        self.decoder_out = nn.Linear(decoder_dim, in_channels * patch_size * patch_size)
        nn.init.xavier_uniform_(self.decoder_out.weight)
        nn.init.zeros_(self.decoder_out.bias)

    def encode(
        self, x_prev: torch.Tensor, x_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode frames to latent actions.

        Args:
            x_prev: Previous frames (B, C, T, H, W)
            x_next: Next frame (B, C, H, W)

        Returns:
            latent_actions: Discrete latent action indices (B, T)
            z_q: Quantized embeddings (B, T, embedding_dim)
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

    def decode(
        self,
        x_prev: torch.Tensor,
        z_q: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent actions to predict next frame.

        Args:
            x_prev: Previous frames (B, C, T, H, W) - will mask all but first
            z_q: Quantized action embeddings (B, T-1, embedding_dim)

        Returns:
            predicted_next_frame: (B, C, H, W)
        """
        B, C, T, H, W = x_prev.shape

        # Mask all frames except the first - forces decoder to use actions
        x_masked = x_prev.clone()
        for t in range(1, T):
            x_masked[:, :, t, :, :] = 0

        # Embed frames
        x = x_masked.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.decoder_patch_embed(x)
        _, C_dec, H_dec, W_dec = x.shape

        x = x.reshape(B, T, H_dec * W_dec, C_dec)
        x = x + self.decoder_pos_embed[:, : H_dec * W_dec, :].unsqueeze(1)

        # Project action embeddings to decoder dimension and add as conditioning
        action_cond = self.decoder_action_proj(z_q)  # (B, T-1, decoder_dim)

        # Add action embeddings to frames (align T-1 actions with T-1 future frames)
        # Frames 1 to T-1 get action conditioning, frame 0 is the prompt
        action_expanded = torch.zeros(B, T, H_dec * W_dec, C_dec, device=x.device)
        action_expanded[:, 1:, :, :] = action_cond.unsqueeze(2)
        x = x + action_expanded

        # Decode
        x = x.reshape(B * T, H_dec * W_dec, C_dec)
        x = self.decoder(x)
        x = x.reshape(B, T, H_dec * W_dec, C_dec)

        # Use only last frame (predicted next frame)
        x = x[:, -1, :, :]  # (B, H_dec*W_dec, C_dec)

        # Project to pixel space
        x = self.decoder_out(x)  # (B, H_dec*W_dec, C*patch_size*patch_size)

        # Reshape to image
        x = x.reshape(B, H_dec, W_dec, C, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, H_dec, patch_size, W_dec, patch_size)
        x = x.reshape(B, C, H_dec * self.patch_size, W_dec * self.patch_size)

        # Resize to original image size if needed
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return torch.sigmoid(x)

    def forward(
        self,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode to get actions, decode to reconstruct.

        Args:
            x_prev: Previous frames (B, C, T, H, W)
            x_next: Next frame (B, C, H, W)

        Returns:
            Dictionary with losses and outputs
        """
        # Encode to get latent actions
        latent_actions, z_q = self.encode(x_prev, x_next)

        # Use actions from timestep 1 to T-1 (T-1 actions for T-1 to predict T)
        # latent_actions has shape (B, T) from encoding T frames
        # We need T-1 actions to predict T-1 future frames
        num_actions = z_q.shape[1] - 1  # T-1
        z_q_for_decode = z_q[:, :num_actions, :] if num_actions > 0 else z_q

        # Decode to reconstruct next frame
        # decode expects x_prev with T frames and z_q with T-1 actions
        reconstructed = self.decode(x_prev, z_q_for_decode)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x_next)

        # Get VQ loss from quantizer
        B = x_prev.shape[0]
        T = x_prev.shape[2]
        z_q_reshaped = z_q.reshape(B * T, -1)  # (B*T, embedding_dim)
        _, _, vq_info = self.vq(z_q_reshaped)
        vq_loss = vq_info["vq_loss"]

        # Variance loss: encourage batch-wise variance in action embeddings
        # This prevents the encoder from collapsing to a single action
        z_for_variance = z_q.detach()  # Detach to only optimize encoder
        action_variance = z_for_variance.var(dim=0).mean()
        variance_loss = (
            -action_variance
        )  # Negative because we want to maximize variance

        return {
            "latent_actions": latent_actions,
            "z_q": z_q,
            "reconstructed": reconstructed,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "variance_loss": variance_loss,
        }


def create_latent_action_model(
    num_frames: int = 16,
    image_size: int = 64,
    in_channels: int = 3,
    encoder_dim: int = 256,
    decoder_dim: int = 512,
    encoder_depth: int = 4,
    decoder_depth: int = 4,
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
        decoder_dim=decoder_dim,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        patch_size=patch_size,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )
