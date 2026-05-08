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

        recon_video, video_indices, tokenizer_loss_dict = self.video_tokenizer(video)

        latent_actions, _ = self.latent_action_model.encode(
            video[:, :, :-1], video[:, :, -1]
        )
        latent_actions = latent_actions[:, : T - 1]

        B_idx, T_idx, H_idx, W_idx = video_indices.shape
        video_tokens = video_indices.reshape(B_idx, T_idx, H_idx * W_idx)

        target_tokens = video_tokens[:, 1:, :]
        dynamics_logits = self.dynamics_model(
            video_tokens[:, :-1, :],
            latent_actions,
            mask_prob=mask_prob,
        )

        B_pred, T_pred, N, V = dynamics_logits.shape
        target_flat = target_tokens.reshape(B_pred * T_pred * N)
        logits_flat = dynamics_logits.reshape(B_pred * T_pred * N, V)
        dynamics_loss = F.cross_entropy(logits_flat, target_flat)

        total_loss = (
            tokenizer_loss_dict["recon_loss"]
            + tokenizer_loss_dict["vq_loss"]
            + dynamics_loss
        )

        return {
            "reconstructed_video": recon_video,
            "video_indices": video_indices,
            "latent_actions": latent_actions,
            "dynamics_logits": dynamics_logits,
            "tokenizer_loss": tokenizer_loss_dict,
            "dynamics_loss": dynamics_loss,
            "total_loss": total_loss,
        }

    def training_step(
        self,
        video: torch.Tensor,
        mask_prob: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Single training step computing all losses.

        Args:
            video: (B, C, T, H, W) input video
            mask_prob: Probability for random masking in dynamics

        Returns:
            Dictionary containing all losses for backpropagation
        """
        return self.forward(video, mask_prob)

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
            use_maskgit: Whether to use MaskGIT sampling (currently uses autoregressive)

        Returns:
            generated_video: (B, C, num_frames, H, W)
        """
        B, C, H, W = prompt_frame.shape

        prompt_frame = prompt_frame.unsqueeze(2).expand(-1, -1, num_frames, -1, -1)

        z_q, prompt_indices, _ = self.video_tokenizer.encode(prompt_frame)

        prompt_tokens = prompt_indices[:, 0, :, :].reshape(B, -1).unsqueeze(1)

        if actions is None:
            actions = torch.randint(
                0,
                self.action_vocab_size,
                (B, num_frames - 1),
                device=prompt_frame.device,
            )

        if use_maskgit and hasattr(self, "sampler"):
            generated_tokens = self._generate_maskgit(
                prompt_tokens,
                actions,
                num_frames,
            )
        else:
            generated_tokens = self.dynamics_model.autoregressive_sample(
                prompt_tokens[:, :1, :],
                actions[:, :1],
                num_frames,
                temperature=2.0,
            )

        z_generated = self.video_tokenizer.decode_indices(generated_tokens)

        generated_video = self.video_tokenizer.decode(z_generated)

        return generated_video

    def _generate_maskgit(
        self,
        prompt_tokens: torch.Tensor,
        actions: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Generate tokens using MaskGIT sampling."""
        B = prompt_tokens.shape[0]
        num_patches = prompt_tokens.shape[2]

        target_tokens = torch.zeros(
            (B, num_frames, num_patches),
            dtype=torch.long,
            device=prompt_tokens.device,
        )
        target_tokens[:, 0, :] = prompt_tokens[:, 0, :]

        mask = torch.ones(
            B, num_frames, num_patches, dtype=torch.bool, device=prompt_tokens.device
        )
        mask[:, :1, :] = False

        max_steps = min(self.sampler.num_steps, 3)
        for step in range(max_steps):
            B_curr, T_curr, N_curr = target_tokens.shape

            logits = self.dynamics_model(
                target_tokens[:, :-1, :],
                actions,
                mask_prob=0.0,
            )

            target_tokens, mask = self.sampler.sample(
                logits.reshape(B_curr, T_curr, N_curr, -1),
                mask,
                step,
            )

        return target_tokens

    def play(
        self,
        current_frame: torch.Tensor,
        action: torch.Tensor,
        current_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Play step - generate next frame given current frame and action.

        Args:
            current_frame: (B, C, H, W) current frame
            action: (B,) latent action indices (int or tensor)
            current_frames: (B, C, T, H, W) history frames, or None for first frame

        Returns:
            next_frame: (B, C, H, W)
        """
        B, C, H, W = current_frame.shape

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=current_frame.device)
        action = action.to(current_frame.device)

        if current_frames is None:
            current_frames = current_frame.unsqueeze(2)

        T_history = current_frames.shape[2]

        z_q, prompt_indices, _ = self.video_tokenizer.encode(current_frames)

        prompt_tokens = prompt_indices.reshape(B, T_history, -1)

        if action.dim() == 0:
            action = action.unsqueeze(0)
        action_expanded = action.unsqueeze(1).expand(-1, T_history)

        next_frame_logits = self.dynamics_model(
            prompt_tokens,
            action_expanded,
            mask_prob=0.0,
        )

        next_frame_logits = next_frame_logits[:, -1, :, :]

        next_token_ids = torch.argmax(next_frame_logits, dim=-1)

        num_patches_per_side = int(next_token_ids.shape[1] ** 0.5)
        next_token_ids_reshaped = next_token_ids.reshape(
            B, num_patches_per_side, num_patches_per_side
        )

        z_next = self.video_tokenizer.decode_indices(
            next_token_ids_reshaped.unsqueeze(1)
        )

        next_frame = self.video_tokenizer.decode(z_next)

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
