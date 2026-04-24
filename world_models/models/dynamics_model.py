import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

from world_models.blocks.st_transformer import STTransformer


class MaskGITSampler:
    """MaskGIT sampling for token-based video generation.

    Uses iterative refinement with a mask schedule to progressively
    reveal tokens during generation.
    """

    def __init__(
        self,
        num_steps: int = 25,
        temperature: float = 2.0,
        mask_schedule: str = "cosine",
    ):
        self.num_steps = num_steps
        self.temperature = temperature
        self.mask_schedule = mask_schedule

    def get_mask_prob(self, step: int) -> float:
        """Get mask probability for given step."""
        if self.mask_schedule == "cosine":
            t = step / self.num_steps
            return 1.0 - (1.0 + math.cos(math.pi * t)) / 2.0
        elif self.mask_schedule == "linear":
            return 1.0 - (step + 1) / self.num_steps
        else:
            raise ValueError(f"Unknown mask schedule: {self.mask_schedule}")

    def sample(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens from logits with mask.

        Args:
            logits: (B, T, vocab_size)
            mask: (B, T) - 1 for tokens to predict, 0 for fixed tokens
            step: Current step in refinement

        Returns:
            sampled_tokens: (B, T)
            new_mask: (B, T)
        """
        probs = F.softmax(logits / self.temperature, dim=-1)

        sampled = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
        sampled = sampled.reshape(mask.shape)

        current_mask_prob = self.get_mask_prob(step)

        keep_mask = mask.float() * (torch.rand_like(mask.float()) > current_mask_prob)
        new_mask = mask * keep_mask

        new_tokens = torch.where(keep_mask.bool(), mask.long(), sampled)

        return new_tokens, new_mask


class DynamicsModel(nn.Module):
    """Dynamics Model for action-controllable video generation.

    A decoder-only transformer that predicts future frame tokens given
    past frame tokens and latent actions. Uses MaskGIT for training
    and sampling.

    Based on Genie paper - uses cross-entropy loss with random masking
    during training, and MaskGIT iterative refinement at inference.
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 64,
        vocab_size: int = 1024,
        embedding_dim: int = 32,
        action_vocab_size: int = 8,
        dim: int = 5120,
        depth: int = 48,
        num_heads: int = 36,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.action_vocab_size = action_vocab_size
        self.dim = dim
        self.patch_size = patch_size

        num_patches = (image_size // patch_size) ** 2

        self.video_embedding = nn.Embedding(vocab_size, dim)
        self.action_embedding = nn.Embedding(action_vocab_size, dim)

        self.video_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, num_patches, dim)
        )
        nn.init.trunc_normal_(self.video_pos_embed, std=0.02)

        self.action_pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        nn.init.trunc_normal_(self.action_pos_embed, std=0.02)

        self.dynamics_transformer = STTransformer(
            num_frames=num_frames,
            num_patches_per_frame=num_patches,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.output_proj = nn.Linear(dim, vocab_size)

    def forward(
        self,
        video_tokens: torch.Tensor,
        actions: torch.Tensor,
        mask_prob: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            video_tokens: (B, T, H*W) - token indices for frames 1 to T
            actions: (B, T) - latent action indices for frames 1 to T
            mask_prob: Probability of masking input tokens

        Returns:
            logits: (B, T, H*W, vocab_size)
        """
        B, T, N = video_tokens.shape

        video_emb = self.video_embedding(video_tokens)

        video_emb = video_emb.reshape(B, T, N, self.dim)

        video_emb = video_emb + self.video_pos_embed[:, :T, :, :]

        action_emb = self.action_embedding(actions)

        action_emb = action_emb + self.action_pos_embed[:, :T, :].unsqueeze(2)

        x = torch.cat(
            [video_emb, action_emb.unsqueeze(2).expand(-1, -1, N, -1)], dim=-1
        )

        x = x.reshape(B, T * N, -1)

        x = self.dynamics_transformer(x)

        x = x.reshape(B, T, N, self.dim)

        logits = self.output_proj(x)

        return logits

    def sample(
        self,
        prompt_tokens: torch.Tensor,
        prompt_actions: torch.Tensor,
        num_frames: int,
        sampler: Optional[MaskGITSampler] = None,
    ) -> torch.Tensor:
        """Sample future frames using MaskGIT.

        Args:
            prompt_tokens: (B, T_prompt, N) - starting frame tokens
            prompt_actions: (B, T_prompt) - actions for prompt frames
            num_frames: Total number of frames to generate
            sampler: MaskGIT sampler instance

        Returns:
            generated_tokens: (B, num_frames, N)
        """
        if sampler is None:
            sampler = MaskGITSampler()

        B, T_prompt, N = prompt_tokens.shape

        all_tokens = [prompt_tokens]
        all_actions = [prompt_actions]

        T_remaining = num_frames - T_prompt

        current_tokens = prompt_tokens
        current_actions = prompt_actions

        for step in range(T_remaining):
            mask = torch.ones(B, N, device=prompt_tokens.device, dtype=torch.long)

            logits = self.forward(current_tokens, current_actions)

            next_token_logits = logits[:, -1, :, :]

            probs = F.softmax(next_token_logits / sampler.temperature, dim=-1)

            next_tokens = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
            next_tokens = next_tokens.reshape(B, N)

            next_action = torch.randint(
                0, self.action_vocab_size, (B,), device=prompt_tokens.device
            )

            current_tokens = torch.cat(
                [current_tokens, next_tokens.unsqueeze(1)], dim=1
            )
            current_actions = torch.cat(
                [current_actions, next_action.unsqueeze(1)], dim=1
            )

            all_tokens.append(next_tokens.unsqueeze(1))
            all_actions.append(next_action.unsqueeze(1))

        generated_tokens = torch.cat(all_tokens, dim=1)

        return generated_tokens

    def autoregressive_sample(
        self,
        prompt_tokens: torch.Tensor,
        prompt_actions: torch.Tensor,
        num_frames: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Simple autoregressive sampling (token by token).

        Args:
            prompt_tokens: (B, T_prompt, N) - starting frame tokens
            prompt_actions: (B, T_prompt) - actions for prompt frames
            num_frames: Total number of frames to generate
            temperature: Sampling temperature

        Returns:
            generated_tokens: (B, num_frames, N)
        """
        B, T_prompt, N = prompt_tokens.shape

        current_tokens = prompt_tokens
        current_actions = prompt_actions

        T_remaining = num_frames - T_prompt

        for step in range(T_remaining):
            logits = self.forward(current_tokens, current_actions)

            next_frame_logits = logits[:, -1, :, :]

            probs = F.softmax(next_frame_logits / temperature, dim=-1)

            next_tokens = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
            next_tokens = next_tokens.reshape(B, N)

            next_action = torch.randint(
                0, self.action_vocab_size, (B,), device=prompt_tokens.device
            )

            current_tokens = torch.cat(
                [current_tokens, next_tokens.unsqueeze(1)], dim=1
            )
            current_actions = torch.cat(
                [current_actions, next_action.unsqueeze(1)], dim=1
            )

        return current_tokens


def create_dynamics_model(
    num_frames: int = 16,
    image_size: int = 64,
    vocab_size: int = 1024,
    embedding_dim: int = 32,
    action_vocab_size: int = 8,
    dim: int = 5120,
    depth: int = 48,
    num_heads: int = 36,
    patch_size: int = 4,
) -> DynamicsModel:
    """Factory function to create a Dynamics Model."""
    return DynamicsModel(
        num_frames=num_frames,
        image_size=image_size,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        action_vocab_size=action_vocab_size,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
    )
