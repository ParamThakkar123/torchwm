import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from world_models.vision.video_tokenizer import VideoTokenizer
from world_models.models.latent_action_model import LatentActionModel
from world_models.models.dynamics_model import DynamicsModel, MaskGITSampler


class Genie(nn.Module):
    """Genie: Generative Interactive Environment.

    A generative model trained from video-only data that can be used as an
    interactive environment. Contains three key components:
    1. Video Tokenizer: Converts raw video frames into discrete tokens
    2. Latent Action Model (LAM): Infers latent actions between frames
    3. Dynamics Model: Predicts future frames given past frames and latent actions

    Based on "Genie: Generative Interactive Environments" paper.
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 64,
        in_channels: int = 3,
        tokenizer_vocab_size: int = 1024,
        tokenizer_embedding_dim: int = 32,
        tokenizer_encoder_dim: int = 256,
        tokenizer_decoder_dim: int = 512,
        action_vocab_size: int = 8,
        action_embedding_dim: int = 32,
        action_encoder_dim: int = 256,
        dynamics_dim: int = 512,
        dynamics_depth: int = 8,
        dynamics_num_heads: int = 8,
        encoder_depth: int = 4,
        decoder_depth: int = 8,
        latent_action_depth: int = 4,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.action_vocab_size = action_vocab_size

        self.video_tokenizer = VideoTokenizer(
            num_frames=num_frames,
            image_size=image_size,
            in_channels=in_channels,
            encoder_dim=tokenizer_encoder_dim,
            decoder_dim=tokenizer_decoder_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=16,
            patch_size=4,
            vocab_size=tokenizer_vocab_size,
            embedding_dim=tokenizer_embedding_dim,
            use_ema=False,
        )

        self.latent_action_model = LatentActionModel(
            num_frames=num_frames,
            image_size=image_size,
            in_channels=in_channels,
            encoder_dim=action_encoder_dim,
            encoder_depth=latent_action_depth,
            num_heads=16,
            patch_size=16,
            vocab_size=action_vocab_size,
            embedding_dim=action_embedding_dim,
            commitment_weight=1.0,
        )

        self.dynamics_model = DynamicsModel(
            num_frames=num_frames,
            image_size=image_size,
            vocab_size=tokenizer_vocab_size,
            embedding_dim=tokenizer_embedding_dim,
            action_vocab_size=action_vocab_size,
            dim=dynamics_dim,
            depth=dynamics_depth,
            num_heads=dynamics_num_heads,
            patch_size=4,
        )

        self.sampler = MaskGITSampler(num_steps=25, temperature=2.0)

    def forward(
        self,
        video: torch.Tensor,
        mask_prob: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through all components.

        Args:
            video: (B, C, T, H, W) input video
            mask_prob: Probability for random masking in dynamics

        Returns:
            Dictionary containing losses and predictions
        """
        B, C, T, H, W = video.shape

        recon_video, video_indices, tokenizer_loss = self.video_tokenizer(video)

        latent_actions, _ = self.latent_action_model.encode(
            video[:, :, :-1], video[:, :, -1]
        )
        latent_actions = latent_actions[:, : T - 1]

        return {
            "reconstructed_video": recon_video,
            "video_indices": video_indices,
            "latent_actions": latent_actions,
            "dynamics_logits": None,
            "tokenizer_loss": tokenizer_loss,
        }

    def generate(
        self,
        prompt_frame: torch.Tensor,
        num_frames: int = 16,
        actions: Optional[torch.Tensor] = None,
        use_maskgit: bool = True,
    ) -> torch.Tensor:
        """Generate video frames given a prompt frame and actions.

        Args:
            prompt_frame: (B, C, H, W) initial frame
            num_frames: Total number of frames to generate
            actions: (B, num_frames-1) latent action indices, or None for random
            use_maskgit: Whether to use MaskGIT sampling

        Returns:
            generated_video: (B, C, T, H, W)
        """
        B, C, H, W = prompt_frame.shape

        prompt_frame = prompt_frame.unsqueeze(2).expand(-1, -1, self.num_frames, -1, -1)

        z_q, prompt_indices, _ = self.video_tokenizer.encode(prompt_frame)

        prompt_tokens = prompt_indices[:, 0, :, :].unsqueeze(1)

        if actions is None:
            actions = torch.randint(
                0,
                self.action_vocab_size,
                (B, num_frames - 1),
                device=prompt_frame.device,
            )

        generated_tokens = self.dynamics_model.autoregressive_sample(
            prompt_tokens[:, :1, :],
            actions[:, :1],
            num_frames,
            temperature=2.0,
        )

        z_generated = self.video_tokenizer.vq.decode_indices(generated_tokens)

        generated_video = self.video_tokenizer.decode(z_generated)

        return generated_video

    def play(
        self,
        prompt_frame: torch.Tensor,
        action: int,
        current_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Play step - generate next frame given current frame and action.

        Args:
            prompt_frame: (B, C, H, W) current frame
            action: (B,) latent action index
            current_frames: (B, C, T, H, W) history frames, or None for first frame

        Returns:
            next_frame: (B, C, H, W)
        """
        B, C, H, W = prompt_frame.shape

        if current_frames is None:
            current_frames = prompt_frame.unsqueeze(2)

        T_history = current_frames.shape[2]

        z_q, prompt_indices, _ = self.video_tokenizer.encode(current_frames)

        prompt_tokens = prompt_indices.reshape(B, T_history, -1)

        action_emb = self.dynamics_model.action_embedding(action)

        x = self.dynamics_model.dynamics_transformer(
            prompt_tokens.reshape(B, T_history * prompt_tokens.shape[2], -1)
        )

        x = x.reshape(B, T_history, -1, self.dynamics_model.dim)

        action_emb_expanded = action_emb.unsqueeze(1).expand(-1, x.shape[2], -1)

        x = x + action_emb_expanded

        next_frame_logits = x[:, -1, :, :]

        next_token_ids = torch.argmax(next_frame_logits, dim=-1)

        z_next = self.video_tokenizer.vq.decode_indices(next_token_ids)

        next_frame = self.video_tokenizer.decode(z_next[:, -1, :, :, :].unsqueeze(1))

        next_frame = next_frame.squeeze(2)

        return next_frame

    def infer_actions(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Infer latent actions from a sequence of frames.

        Args:
            frames: (B, C, T, H, W) video frames

        Returns:
            latent_actions: (B, T-1) inferred latent action indices
        """
        latent_actions, _ = self.latent_action_model.encode(
            frames[:, :, :-1], frames[:, :, -1]
        )
        return latent_actions

    def get_num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_genie(
    num_frames: int = 16,
    image_size: int = 64,
    in_channels: int = 3,
    tokenizer_vocab_size: int = 1024,
    tokenizer_embedding_dim: int = 32,
    action_vocab_size: int = 8,
    action_embedding_dim: int = 32,
    dynamics_dim: int = 5120,
    dynamics_depth: int = 48,
    dynamics_num_heads: int = 36,
) -> Genie:
    """Factory function to create a Genie model."""
    return Genie(
        num_frames=num_frames,
        image_size=image_size,
        in_channels=in_channels,
        tokenizer_vocab_size=tokenizer_vocab_size,
        tokenizer_embedding_dim=tokenizer_embedding_dim,
        action_vocab_size=action_vocab_size,
        action_embedding_dim=action_embedding_dim,
        dynamics_dim=dynamics_dim,
        dynamics_depth=dynamics_depth,
        dynamics_num_heads=dynamics_num_heads,
    )


def create_genie_small(
    num_frames: int = 16,
    image_size: int = 64,
) -> Genie:
    """Create a smaller Genie model for development/testing."""
    return Genie(
        num_frames=num_frames,
        image_size=image_size,
        tokenizer_vocab_size=1024,
        tokenizer_embedding_dim=32,
        action_vocab_size=8,
        action_embedding_dim=32,
        dynamics_dim=512,
        dynamics_depth=8,
        dynamics_num_heads=8,
        encoder_depth=4,
        decoder_depth=8,
        latent_action_depth=8,
    )


def create_genie_large(
    num_frames: int = 16,
    image_size: int = 64,
) -> Genie:
    """Create the full 11B parameter Genie model (approximate)."""
    return Genie(
        num_frames=num_frames,
        image_size=image_size,
        tokenizer_vocab_size=1024,
        tokenizer_embedding_dim=32,
        action_vocab_size=8,
        action_embedding_dim=32,
        dynamics_dim=5120,
        dynamics_depth=48,
        dynamics_num_heads=36,
        encoder_depth=12,
        decoder_depth=20,
        latent_action_depth=20,
    )
