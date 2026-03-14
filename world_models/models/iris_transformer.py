import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class IRISTransformer(nn.Module):
    """Autoregressive Transformer for world modeling.

    Models the dynamics of the environment by predicting:
    - Next frame tokens (transition model)
    - Rewards
    - Episode termination

    The Transformer operates on sequences of interleaved frame tokens and actions.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        tokens_per_frame: int = 16,
        action_size: int = 18,  # Number of Atari actions
        embed_dim: int = 256,
        num_layers: int = 10,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.tokens_per_frame = tokens_per_frame
        self.action_size = action_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.action_embedding = nn.Embedding(action_size, embed_dim)

        # Positional embeddings
        # Max sequence length: (tokens_per_frame + 1) * timesteps
        # 16 tokens + 1 action per timestep = 17 tokens/timestep
        max_tokens = tokens_per_frame + 1  # tokens + action
        max_seq_len = max_tokens * 50  # Support up to 50 timesteps
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Token prediction head (for next frame tokens)
        # Predicts each token of the next frame
        self.token_head = nn.Linear(embed_dim, vocab_size)

        # Reward prediction head
        self.reward_head = nn.Linear(embed_dim, 1)

        # Termination prediction head
        self.termination_head = nn.Linear(
            embed_dim, 2
        )  # Binary: 0=continue, 1=terminal

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.action_embedding.weight, std=0.02)

        # Apply special initialization to output heads
        nn.init.zeros_(self.token_head.bias)
        nn.init.zeros_(self.reward_head.bias)
        nn.init.zeros_(self.termination_head.bias)

    def _build_sequence(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Build the interleaved token-action sequence.

        Args:
            tokens: Frame tokens (B, T, K)
            actions: Actions (B, T)

        Returns:
            Sequence ready for transformer (B, T*(K+1), embed_dim)
        """
        B, T, K = tokens.shape

        tokens_flat = tokens.reshape(B, T * K)
        token_embeds = self.token_embedding(tokens_flat)

        action_embeds = self.action_embedding(actions)
        token_embeds = token_embeds.reshape(B, T, K, self.embed_dim)

        action_embeds_expanded = action_embeds.unsqueeze(2)
        sequence = torch.cat([token_embeds, action_embeds_expanded], dim=2)

        sequence = sequence.reshape(B, T * (K + 1), self.embed_dim)
        sequence = sequence + self.pos_embedding[:, : T * (K + 1), :]

        return sequence

    def forward(
        self,
        tokens: torch.Tensor,  # (B, T, K) - frame tokens
        actions: torch.Tensor,  # (B, T) - actions
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the Transformer world model.

        Args:
            tokens: Frame tokens (B, T, K) where T is timesteps
            actions: Actions (B, T)
            mask: Optional attention mask

        Returns:
            token_logits: Next token predictions (B, T, K, vocab_size)
            rewards: Predicted rewards (B, T)
            terminations: Predicted terminations (B, T, 2)
        """
        B, T, K = tokens.shape

        # Flatten tokens: (B, T, K) -> (B, T*K)
        tokens_flat = tokens.reshape(B, T * K)

        # Embed tokens
        token_embeds = self.token_embedding(tokens_flat)  # (B, T*K, embed_dim)

        # Embed actions: (B, T) -> (B, T, embed_dim)
        action_embeds = self.action_embedding(actions)  # (B, T, embed_dim)

        # Reshape token embeddings to (B, T, K, embed_dim)
        token_embeds = token_embeds.reshape(B, T, K, self.embed_dim)

        # Interleave: for each timestep, concat token embeddings with action embedding
        # sequence: [tokens_t, action_t] for each t
        # Result: (B, T, K+1, embed_dim)
        action_embeds_expanded = action_embeds.unsqueeze(2)  # (B, T, 1, embed_dim)
        sequence = torch.cat(
            [token_embeds, action_embeds_expanded], dim=2
        )  # (B, T, K+1, embed_dim)

        # Flatten: (B, T, K+1, embed_dim) -> (B, T*(K+1), embed_dim)
        sequence = sequence.reshape(B, T * (K + 1), self.embed_dim)

        # Create position ids
        (torch.arange(T * (K + 1), device=tokens.device).unsqueeze(0).expand(B, -1))

        # Add positional embeddings
        sequence = sequence + self.pos_embedding[:, : T * (K + 1), :]

        # Apply transformer
        hidden = self.transformer(sequence, mask=mask)
        hidden = self.layer_norm(hidden)

        # Reshape hidden states back to per-timestep structure
        # Each timestep has K tokens + 1 action = K+1 positions
        # hidden[:, i*(K+1):i*(K+1)+K, :] = frame token predictions for step i
        # hidden[:, i*(K+1)+K, :] = action token predictions for step i

        # Reshape to (B, T, K+1, embed_dim)
        hidden = hidden.reshape(B, T, K + 1, self.embed_dim)

        # Extract predictions for each timestep
        token_hidden = hidden[:, :, :K, :]  # (B, T, K, embed_dim)
        action_hidden = hidden[:, :, K, :]  # (B, T, embed_dim)

        # Token predictions (next frame)
        token_logits = self.token_head(token_hidden)  # (B, T, K, vocab_size)

        # Reward predictions (from action position)
        rewards = self.reward_head(action_hidden).squeeze(-1)  # (B, T)

        # Termination predictions
        terminations = self.termination_head(action_hidden)  # (B, T, 2)

        return token_logits, rewards, terminations

    def predict_next_tokens(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the next frame tokens autoregressively.

        Used during imagination rollouts.

        Args:
            tokens: Current frame tokens (B, K)
            actions: Actions taken (B,)

        Returns:
            token_logits: Next frame token predictions (B, K, vocab_size)
            action_hidden: Hidden states for reward prediction (B, embed_dim)
        """
        # Handle token shapes: (B, H, W) -> (B, K) -> (B, 1, K)
        if tokens.dim() == 3:
            # tokens is (B, H, W) grid of tokens, flatten to (B, K)
            B_grid, H, W = tokens.shape
            tokens = tokens.reshape(B_grid, H * W)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)  # (B, K) -> (B, 1, K)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)  # (B,) -> (B, 1)

        token_logits, _, _ = self.forward(tokens, actions)

        # Get action hidden states for reward prediction
        B, T, K, embed_dim = token_logits.shape
        hidden = self.layer_norm(
            self.transformer(self._build_sequence(tokens, actions), mask=None)
        )
        hidden = hidden.reshape(B, T, K + 1, self.embed_dim)
        action_hidden = hidden[:, -1, K, :]  # (B, embed_dim)

        return (
            token_logits[:, -1, :, :],
            action_hidden,
        )  # Return last timestep predictions

    def sample_next_tokens(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample next tokens from the distribution.

        Args:
            tokens: Current frame tokens (B, K)
            actions: Actions taken (B,)
            temperature: Sampling temperature (higher = more random)

        Returns:
            sampled_tokens: Sampled token indices (B, K)
            log_probs: Log probabilities of sampled tokens (B, K)
        """
        token_logits, _ = self.predict_next_tokens(tokens, actions)

        # Apply temperature
        token_logits = token_logits / temperature

        # Sample from categorical
        probs = F.softmax(token_logits, dim=-1)
        sampled_indices = torch.multinomial(probs.reshape(-1, self.vocab_size), 1)
        sampled_indices = sampled_indices.reshape_as(tokens)

        # Compute log probabilities
        log_probs = F.log_softmax(token_logits, dim=-1)
        log_probs = torch.gather(log_probs, -1, sampled_indices.unsqueeze(-1)).squeeze(
            -1
        )

        return sampled_indices, log_probs


class IRISWorldModel(nn.Module):
    """Complete IRIS World Model combining autoencoder and transformer.

    This is the core component that learns environment dynamics entirely
    in the "imaginary" latent space.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        transformer: IRISTransformer,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.transformer = transformer

    def forward(
        self,
        observations: torch.Tensor,  # (B, T+1, C, H, W)
        actions: torch.Tensor,  # (B, T)
    ) -> Tuple[dict, dict]:
        """Full world model forward pass.

        Args:
            observations: Image sequence (B, T+1, C, H, W)
            actions: Actions (B, T)

        Returns:
            predictions: Dictionary with predicted tokens, rewards, terminations
            losses: Dictionary with loss components
        """
        B, T_plus_1, C, H, W = observations.shape
        T = T_plus_1 - 1

        # Encode each frame to tokens
        tokens_list = []
        for t in range(T_plus_1):
            obs_t = observations[:, t]  # (B, C, H, W)
            _, indices_t, _ = self.encoder(obs_t)
            tokens_list.append(indices_t)

        # Stack tokens: (B, T+1, K)
        tokens = torch.stack(tokens_list, dim=1)

        # Get transformer predictions
        token_logits, rewards_pred, terminations_pred = self.transformer(
            tokens[:, :-1],  # (B, T, K) - all frames except last
            actions,  # (B, T)
        )

        # Decode predictions to images (for visualization)
        # For each timestep, decode the predicted tokens
        decoded_frames = []
        for t in range(T):
            next_tokens_pred = token_logits[:, t, :, :].argmax(dim=-1)  # Greedy
            decoded_frames.append(self.decoder.decode_from_embeddings(next_tokens_pred))

        decoded_frames = torch.stack(decoded_frames, dim=1) if decoded_frames else None

        # Get actual next tokens for loss computation
        next_tokens = tokens[:, 1:]  # (B, T, K)

        # Compute losses
        token_loss = F.cross_entropy(
            token_logits.reshape(-1, self.transformer.vocab_size),
            next_tokens.reshape(-1),
            reduction="mean",
        )

        # Reward and termination losses would be computed with actual labels
        # (These are computed in the training loop)

        predictions = {
            "token_logits": token_logits,
            "rewards": rewards_pred,
            "terminations": terminations_pred,
            "decoded_frames": decoded_frames,
        }

        losses = {
            "token_loss": token_loss,
        }

        return predictions, losses

    def imagine(
        self,
        initial_tokens: torch.Tensor,  # (B, K)
        policy: nn.Module,
        horizon: int = 20,
        temperature: float = 1.0,
    ) -> dict:
        """Generate imagined trajectories.

        Args:
            initial_tokens: Initial frame tokens (B, K)
            policy: Policy network to sample actions
            horizon: Number of steps to imagine
            temperature: Sampling temperature for token prediction

        Returns:
            imagined: Dictionary with imagined trajectories
        """

        # Lists to store trajectory
        tokens_history = [initial_tokens]
        actions_history = []
        rewards_history = []
        terminations_history = []

        # Get initial reconstruction for policy input
        current_tokens = initial_tokens

        for step in range(horizon):
            # Get action from policy (using decoded frame)
            with torch.no_grad():
                decoded_frame = self.decoder.decode_from_embeddings(current_tokens)
                action = policy(decoded_frame)
                actions_history.append(action)

            # Predict next tokens
            sampled_tokens, log_probs = self.transformer.sample_next_tokens(
                current_tokens,
                action.squeeze(-1) if action.dim() > 1 else action,
                temperature,
            )

            # Get reward and termination predictions
            with torch.no_grad():
                token_logits, action_hidden = self.transformer.predict_next_tokens(
                    current_tokens, action
                )
                reward = self.transformer.reward_head(action_hidden).mean()
                termination_logits = self.transformer.termination_head(action_hidden)
                termination = torch.softmax(termination_logits, dim=-1)[:, 1]

            tokens_history.append(sampled_tokens)
            rewards_history.append(reward)
            terminations_history.append(termination)

            # Update current tokens
            current_tokens = sampled_tokens

            # Early stopping if terminal
            if termination.mean() > 0.5:
                break

        return {
            "tokens": torch.stack(tokens_history, dim=1),  # (B, H+1, K)
            "actions": torch.stack(actions_history, dim=1) if actions_history else None,
            "rewards": torch.stack(rewards_history, dim=1) if rewards_history else None,
            "terminations": (
                torch.stack(terminations_history, dim=1)
                if terminations_history
                else None
            ),
        }
