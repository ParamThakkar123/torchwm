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

    Based on "Genie: Generative Interactive Environments" paper (arXiv:2402.15391).

    Training follows two phases as per paper:
    1. Train video tokenizer first (on video tokens)
    2. Co-train LAM (from pixels) and dynamics model (on video tokens)

    The LAM uses VQ-VAE training with:
    - Encoder: Takes x1:t and x_{t+1} → outputs latent actions
    - Decoder: Takes x1:t-1 (masked) + actions → reconstructs x_t
    - Auxiliary variance loss to prevent action collapse

    At inference, latent actions are stopgrad'd when passed to dynamics model.
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 64,
        in_channels: int = 3,
        tokenizer_vocab_size: int = 1024,
        tokenizer_embedding_dim: int = 32,
        tokenizer_encoder_dim: int = 512,
        tokenizer_decoder_dim: int = 1024,
        action_vocab_size: int = 8,
        action_embedding_dim: int = 32,
        action_encoder_dim: int = 1024,
        action_decoder_dim: int = 1024,
        dynamics_dim: int = 5120,
        dynamics_depth: int = 48,
        dynamics_num_heads: int = 36,
        encoder_depth: int = 12,
        decoder_depth: int = 20,
        latent_action_depth: int = 20,
        use_bfloat16: bool = False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.action_vocab_size = action_vocab_size
        self.use_bfloat16 = use_bfloat16

        # Video Tokenizer (VQ-VAE with ST-Transformer)
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

        # Latent Action Model (VQ-VAE with ST-Transformer decoder)
        self.latent_action_model = LatentActionModel(
            num_frames=num_frames,
            image_size=image_size,
            in_channels=in_channels,
            encoder_dim=action_encoder_dim,
            decoder_dim=action_decoder_dim,
            encoder_depth=latent_action_depth,
            decoder_depth=latent_action_depth,
            num_heads=16,
            patch_size=16,
            vocab_size=action_vocab_size,
            embedding_dim=action_embedding_dim,
            commitment_weight=1.0,
        )

        # Dynamics Model (MaskGIT transformer)
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

        # MaskGIT sampler for inference
        self.sampler = MaskGITSampler(num_steps=25, temperature=2.0)

    def forward(
        self,
        video: torch.Tensor,
        mask_prob: float = 0.5,
        training_phase: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through all components.

        Args:
            video: (B, C, T, H, W) input video
            mask_prob: Probability for random masking in dynamics (0.5-1.0)
            training_phase: "all", "tokenizer", or "lam_dynamics"

        Returns:
            Dictionary containing losses and predictions
        """
        B, C, T, H, W = video.shape

        if training_phase == "tokenizer":
            # Phase 1: Train only video tokenizer
            recon_video, video_indices, tokenizer_loss_dict = self.video_tokenizer(
                video
            )
            return {
                "reconstructed_video": recon_video,
                "video_indices": video_indices,
                "tokenizer_loss": tokenizer_loss_dict,
                "vq_loss": tokenizer_loss_dict["vq_loss"],
                "total_loss": tokenizer_loss_dict["recon_loss"]
                + tokenizer_loss_dict["vq_loss"],
            }

        # Phase 2 or 3: Get video tokens first (frozen or training)
        with (
            torch.no_grad() if training_phase == "lam_dynamics" else torch.enable_grad()
        ):
            recon_video, video_indices, tokenizer_loss_dict = self.video_tokenizer(
                video
            )
            video_tokens = video_indices.reshape(B, T, -1)  # (B, T, H*W)

        # ===== LATENT ACTION MODEL =====
        # Train LAM from pixels - includes encoder + decoder losses
        lam_output = self.latent_action_model(
            video[:, :, :-1],  # x1:T-1
            video[:, :, -1],  # x_T
        )

        # Get latent actions - apply stopgrad for dynamics (as per paper)
        latent_actions = lam_output["latent_actions"]  # (B, T-1)
        z_q = lam_output["z_q"]  # (B, T-1, embedding_dim)

        # stopgrad on latent actions when passing to dynamics (per paper Section 2.1)
        z_q_for_dynamics = z_q.detach()

        # Map z_q to action indices for dynamics model
        # z_q is (B, T-1, embedding_dim), we need (B, T-1) indices
        # Use the latent_actions directly
        actions_for_dynamics = latent_actions[:, : T - 1]

        # ===== DYNAMICS MODEL =====
        # Predict next frame tokens given past tokens and latent actions
        # Input: video_tokens[:, :-1] (past frames), actions_for_dynamics
        # Target: video_tokens[:, 1:] (next frames)
        target_tokens = video_tokens[:, 1:, :]  # (B, T-1, H*W)

        dynamics_logits = self.dynamics_model(
            video_tokens[:, :-1, :],
            actions_for_dynamics,
            mask_prob=mask_prob,
        )

        # Compute dynamics loss
        B_pred, T_pred, N, V = dynamics_logits.shape
        target_flat = target_tokens.reshape(B_pred * T_pred * N)
        logits_flat = dynamics_logits.reshape(B_pred * T_pred * N, V)
        dynamics_loss = F.cross_entropy(logits_flat, target_flat)

        # ===== TOTAL LOSS =====
        # According to paper: co-train LAM and dynamics
        # LAM losses: recon_loss (from decoder) + vq_loss + variance_loss
        # Dynamics loss: cross-entropy on token prediction

        lam_recon_loss = lam_output["recon_loss"]
        lam_vq_loss = lam_output["vq_loss"]
        lam_variance_loss = lam_output["variance_loss"]

        total_loss = lam_recon_loss + lam_vq_loss + lam_variance_loss + dynamics_loss

        return {
            "reconstructed_video": recon_video,
            "video_indices": video_indices,
            "latent_actions": latent_actions,
            "lam_reconstructed": lam_output["reconstructed"],
            "dynamics_logits": dynamics_logits,
            "tokenizer_loss": tokenizer_loss_dict,
            "vq_loss": tokenizer_loss_dict["vq_loss"],
            "lam_recon_loss": lam_recon_loss,
            "lam_vq_loss": lam_vq_loss,
            "lam_variance_loss": lam_variance_loss,
            "dynamics_loss": dynamics_loss,
            "total_loss": total_loss,
        }

    def training_step(
        self,
        video: torch.Tensor,
        mask_prob: float = 0.5,
        training_phase: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """Single training step computing all losses.

        Args:
            video: (B, C, T, H, W) input video
            mask_prob: Probability for random masking in dynamics
            training_phase: "all", "tokenizer", or "lam_dynamics"

        Returns:
            Dictionary containing all losses for backpropagation
        """
        if self.use_bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                return self.forward(video, mask_prob, training_phase)
        return self.forward(video, mask_prob, training_phase)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to discrete tokens.

        Args:
            video: (B, C, T, H, W)

        Returns:
            video_tokens: (B, T, H*W)
        """
        _, video_indices, _ = self.video_tokenizer(video)
        return video_indices.reshape(video_indices.shape[0], video_indices.shape[1], -1)

    def infer_actions(self, frames: torch.Tensor) -> torch.Tensor:
        """Infer latent actions from a sequence of frames.

        Args:
            frames: (B, C, T, H, W) video frames

        Returns:
            latent_actions: (B, T-1) inferred latent action indices
        """
        lam_output = self.latent_action_model(
            frames[:, :, :-1],
            frames[:, :, -1],
        )
        return lam_output["latent_actions"]

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
            generated_video: (B, C, num_frames, H, W)
        """
        B, C, H, W = prompt_frame.shape

        # Tokenize prompt frame
        prompt_frame_expanded = prompt_frame.unsqueeze(2).expand(
            -1, -1, num_frames, -1, -1
        )
        _, prompt_indices, _ = self.video_tokenizer(prompt_frame_expanded)

        # Use first frame tokens as prompt
        prompt_tokens = prompt_indices[:, 0, :, :].reshape(B, -1).unsqueeze(1)

        # Sample random actions if not provided
        if actions is None:
            actions = torch.randint(
                0,
                self.action_vocab_size,
                (B, num_frames - 1),
                device=prompt_frame.device,
            )

        # Generate
        if use_maskgit and hasattr(self, "sampler"):
            generated_tokens = self._generate_maskgit(
                prompt_tokens, actions, num_frames
            )
        else:
            generated_tokens = self.dynamics_model.autoregressive_sample(
                prompt_tokens[:, :1, :],
                actions[:, :1],
                num_frames,
                temperature=2.0,
            )

        # Decode tokens to video
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
            logits = self.dynamics_model(
                target_tokens[:, :-1, :],
                actions,
                mask_prob=0.0,
            )

            target_tokens, mask = self.sampler.sample(
                logits.reshape(B, num_frames, num_patches, -1),
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
            action: (B,) latent action indices
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

        # Tokenize current frames
        _, prompt_indices, _ = self.video_tokenizer(current_frames)
        prompt_tokens = prompt_indices.reshape(B, T_history, -1)

        if action.dim() == 0:
            action = action.unsqueeze(0)
        action_expanded = action.unsqueeze(1).expand(-1, T_history)

        # Predict next frame
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

        return next_frame.squeeze(2)

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
    use_bfloat16: bool = False,
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
        use_bfloat16=use_bfloat16,
    )


def create_genie_small(
    num_frames: int = 16,
    image_size: int = 64,
    use_bfloat16: bool = False,
) -> Genie:
    """Create a smaller Genie model for development/testing."""
    return Genie(
        num_frames=num_frames,
        image_size=image_size,
        tokenizer_vocab_size=1024,
        tokenizer_embedding_dim=32,
        tokenizer_encoder_dim=256,
        tokenizer_decoder_dim=512,
        action_vocab_size=8,
        action_embedding_dim=32,
        action_encoder_dim=512,
        action_decoder_dim=512,
        dynamics_dim=512,
        dynamics_depth=8,
        dynamics_num_heads=8,
        encoder_depth=4,
        decoder_depth=8,
        latent_action_depth=8,
        use_bfloat16=use_bfloat16,
    )


def create_genie_large(
    num_frames: int = 16,
    image_size: int = 64,
    use_bfloat16: bool = True,
) -> Genie:
    """Create the full 11B parameter Genie model (approximate)."""
    return Genie(
        num_frames=num_frames,
        image_size=image_size,
        tokenizer_vocab_size=1024,
        tokenizer_embedding_dim=32,
        tokenizer_encoder_dim=512,
        tokenizer_decoder_dim=1024,
        action_vocab_size=8,
        action_embedding_dim=32,
        action_encoder_dim=1024,
        action_decoder_dim=1024,
        dynamics_dim=5120,
        dynamics_depth=48,
        dynamics_num_heads=36,
        encoder_depth=12,
        decoder_depth=20,
        latent_action_depth=20,
        use_bfloat16=use_bfloat16,
    )
