import torch
import torch.nn as nn
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Convolutional block with adaptive group normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        stride: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, stride, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        self.cond_embed = nn.Linear(cond_dim, out_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)

        scale, bias = self.cond_embed(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)

        return self.act(x * (1 + scale) + bias)


class RewardTerminationModel(nn.Module):
    """
    Reward and termination prediction model.
    CNN + LSTM architecture following DIAMOND paper specifications.

    Args:
        obs_channels: Number of observation channels (3 for RGB)
        action_dim: Number of possible actions
        channels: List of channel sizes for conv blocks
        lstm_dim: LSTM hidden dimension
        cond_dim: Conditioning dimension for adaptive norm
    """

    def __init__(
        self,
        obs_channels: int = 3,
        action_dim: int = 18,
        channels: Tuple[int, ...] = (32, 32, 32, 32),
        lstm_dim: int = 512,
        cond_dim: int = 128,
    ):
        super().__init__()
        self.obs_channels = obs_channels
        self.action_dim = action_dim
        self.lstm_dim = lstm_dim

        self.action_embed = nn.Embedding(action_dim, cond_dim)

        self.conv_blocks = nn.ModuleList()
        in_ch = obs_channels
        for i, out_ch in enumerate(channels):
            self.conv_blocks.append(ConvBlock(in_ch, out_ch, cond_dim, stride=2))
            in_ch = out_ch

        self.lstm = nn.LSTM(
            input_size=channels[-1],
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True,
        )

        self.reward_head = nn.Linear(lstm_dim, 3)
        self.termination_head = nn.Linear(lstm_dim, 2)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of reward/termination model.

        Args:
            obs: Observations [B, T, C, H, W]
            actions: Actions [B, T]
            hidden_state: Optional (h, c) hidden states

        Returns:
            reward_logits: Reward predictions [B, T, 3] (for -1, 0, 1)
            termination_logits: Termination predictions [B, T, 2]
            hidden_state: Updated (h, c) hidden states
        """
        B, T, C, H, W = obs.shape

        obs_flat = obs.view(B * T, C, H, W)
        actions_flat = actions.view(B * T)

        action_emb = self.action_embed(actions_flat)

        h = obs_flat
        for conv_block in self.conv_blocks:
            h = conv_block(h, action_emb)

        h = h.mean(dim=[2, 3])
        h = h.view(B, T, -1)

        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(h)
        else:
            lstm_out, hidden_state = self.lstm(h, hidden_state)

        reward_logits = self.reward_head(lstm_out)
        termination_logits = self.termination_head(lstm_out)

        return reward_logits, termination_logits, hidden_state

    def predict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, bool, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict reward and termination for a single step.

        Args:
            obs: Single observation [B, C, H, W]
            actions: Single action [B]
            hidden_state: Optional (h, c) hidden states

        Returns:
            reward: Predicted reward (as class index 0,1,2 -> -1,0,1)
            terminated: Predicted termination as boolean
            hidden_state: Updated (h, c) hidden states
        """
        obs = obs.unsqueeze(1)
        actions = actions.unsqueeze(1)

        reward_logits, term_logits, hidden_state = self.forward(
            obs, actions, hidden_state
        )

        reward = reward_logits.argmax(dim=-1) - 1
        terminated = term_logits.argmax(dim=-1).bool()

        return reward.squeeze(-1).float(), terminated.squeeze(-1), hidden_state

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden states."""
        h = torch.zeros(1, batch_size, self.lstm_dim, device=device)
        c = torch.zeros(1, batch_size, self.lstm_dim, device=device)
        return (h, c)


class RewardTerminationLoss(nn.Module):
    """Loss function for reward and termination prediction."""

    def __init__(self):
        super().__init__()
        self.reward_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.termination_criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        reward_logits: torch.Tensor,
        termination_logits: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss for reward and termination predictions.

        Args:
            reward_logits: [B, T, 3]
            termination_logits: [B, T, 2]
            rewards: Rewards as class indices [B, T] (values -1, 0, 1 mapped to 0, 1, 2)
            terminated: Termination flags [B, T]

        Returns:
            total_loss, reward_loss, termination_loss
        """
        reward_targets = (rewards + 1).long()

        # use reshape to avoid issues when tensors are non-contiguous
        reward_loss = self.reward_criterion(
            reward_logits.reshape(-1, 3), reward_targets.view(-1)
        )
        termination_loss = self.termination_criterion(
            termination_logits.reshape(-1, 2), terminated.long().view(-1)
        )

        total_loss = reward_loss + termination_loss

        return total_loss, reward_loss, termination_loss
