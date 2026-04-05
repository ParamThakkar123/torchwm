import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from world_models.vision.vq_layer import VectorQuantizer, VectorQuantizerEMA
from world_models.blocks.st_transformer import STTransformer


class VideoTokenizer(nn.Module):
    """Video Tokenizer using VQ-VAE with ST-Transformer.

    Compresses video frames into discrete tokens using a VQ-VAE objective.
    Uses spatiotemporal transformer in both encoder and decoder for improved
    temporal dynamics encoding.

    Based on Genie paper: "Neural Discrete Representation Learning" (VQ-VAE)
    and "Spatiotemporal Transformer" architecture.
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 64,
        in_channels: int = 3,
        encoder_dim: int = 512,
        decoder_dim: int = 1024,
        encoder_depth: int = 12,
        decoder_depth: int = 20,
        num_heads: int = 16,
        patch_size: int = 4,
        vocab_size: int = 1024,
        embedding_dim: int = 32,
        commitment_weight: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        num_patches = (image_size // patch_size) ** 2

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.patch_embed = nn.Conv2d(
            in_channels,
            encoder_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames * num_patches, encoder_dim)
        )
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)

        self.encoder = STTransformer(
            num_frames=num_frames,
            num_patches_per_frame=num_patches,
            dim=encoder_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        self.to_vq_embedding = nn.Linear(encoder_dim, embedding_dim)

        if use_ema:
            self.vq = VectorQuantizerEMA(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                commitment_weight=commitment_weight,
                ema_decay=ema_decay,
            )
        else:
            self.vq = VectorQuantizer(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                commitment_weight=commitment_weight,
            )

        self.from_vq_embedding = nn.Linear(embedding_dim, decoder_dim)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames * num_patches, decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder = STTransformer(
            num_frames=num_frames,
            num_patches_per_frame=num_patches,
            dim=decoder_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        self.decoder_proj = nn.ConvTranspose2d(
            decoder_dim,
            in_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Encode video to discrete tokens.

        Args:
            x: Video tensor (B, C, T, H, W)

        Returns:
            z_q: Quantized embeddings (B, T, H', W', embedding_dim)
            indices: Token indices (B, T, H', W')
            vq_loss: Dictionary with VQ loss components
        """
        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        x = self.patch_embed(x)
        _, C_enc, H_enc, W_enc = x.shape

        x = x.reshape(B, T, C_enc, H_enc, W_enc)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, T * H_enc * W_enc, C_enc)

        seq_len = T * H_enc * W_enc
        pos_embed = self.encoder_pos_embed[:, :seq_len, :]
        x = x + pos_embed

        x = self.encoder(x)

        x = x.reshape(B, T, H_enc, W_enc, self.encoder_dim)

        z_all = []
        indices_all = []
        vq_loss_all = {}

        for t in range(T):
            x_t = x[:, t, :, :, :]
            x_t = x_t.permute(0, 3, 1, 2).reshape(B, self.encoder_dim, H_enc * W_enc)
            x_t = x_t.permute(0, 2, 1)

            x_t_embed = self.to_vq_embedding(x_t)
            x_t_embed = x_t_embed.reshape(B, H_enc, W_enc, self.embedding_dim)
            x_t_embed = x_t_embed.permute(0, 3, 1, 2)

            z_q_t, indices_t, vq_loss_t = self.vq(x_t_embed)
            z_all.append(z_q_t.reshape(B, H_enc, W_enc, self.embedding_dim))
            indices_all.append(indices_t.reshape(B, H_enc, W_enc))
            for k, v in vq_loss_t.items():
                if k not in vq_loss_all:
                    vq_loss_all[k] = []
                vq_loss_all[k].append(v)

        z_q = torch.stack(z_all, dim=1)
        indices = torch.stack(indices_all, dim=1)

        vq_loss = {}
        for k, v in vq_loss_all.items():
            stacked = torch.stack(v)
            if k == "perplexity":
                vq_loss[k] = stacked.mean()
            else:
                vq_loss[k] = stacked.mean()

        return z_q, indices, vq_loss

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens to video frames.

        Args:
            z_q: Quantized embeddings (B, T, H', W', embedding_dim)

        Returns:
            Reconstructed video (B, C, T, H, W)
        """
        B, T, H_dec, W_dec, _ = z_q.shape

        x = z_q.reshape(B, T * H_dec * W_dec, -1)

        x = self.from_vq_embedding(x)

        seq_len = T * H_dec * W_dec
        pos_embed = self.decoder_pos_embed[:, :seq_len, :]
        x = x + pos_embed

        x = self.decoder(x)

        x = x.reshape(B, T, H_dec, W_dec, self.decoder_dim)
        x = x.permute(0, 4, 1, 2, 3).reshape(B * T, self.decoder_dim, H_dec, W_dec)

        x = self.decoder_proj(x)

        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Full forward pass with VQ-VAE objective.

        Args:
            x: Video tensor (B, C, T, H, W)

        Returns:
            reconstructed: Reconstructed video (B, C, T, H, W)
            indices: Token indices (B, T, H', W')
            loss_dict: Dictionary containing loss components
        """
        z_q, indices, vq_loss = self.encode(x)

        reconstructed = self.decode(z_q)

        recon_loss = F.mse_loss(reconstructed, x)

        loss_dict = {
            "recon_loss": recon_loss,
            "vq_loss": vq_loss["vq_loss"],
            "perplexity": vq_loss["perplexity"],
        }

        return reconstructed, indices, loss_dict


def create_video_tokenizer(
    num_frames: int = 16,
    image_size: int = 64,
    in_channels: int = 3,
    encoder_dim: int = 512,
    decoder_dim: int = 1024,
    encoder_depth: int = 12,
    decoder_depth: int = 20,
    num_heads: int = 16,
    patch_size: int = 4,
    vocab_size: int = 1024,
    embedding_dim: int = 32,
    use_ema: bool = False,
) -> VideoTokenizer:
    """Factory function to create a Video Tokenizer."""
    return VideoTokenizer(
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
        use_ema=use_ema,
    )
