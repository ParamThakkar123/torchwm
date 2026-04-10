import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for DIAMOND RL training.
    Shared CNN-LSTM trunk with separate policy and value heads.
    """

    def __init__(
        self,
        obs_channels: int = 3,
        action_dim: int = 18,
        channels: Tuple[int, ...] = (32, 32, 64, 64),
        lstm_dim: int = 512,
    ):
        super().__init__()
        self.obs_channels = obs_channels
        self.action_dim = action_dim

        self.conv_blocks = nn.ModuleList()
        in_ch = obs_channels
        for i, out_ch in enumerate(channels):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                )
            )
            in_ch = out_ch

        self.lstm = nn.LSTM(
            input_size=channels[-1],
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True,
        )

        self.policy_head = nn.Linear(lstm_dim, action_dim)
        self.value_head = nn.Linear(lstm_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of actor-critic network.

        Args:
            obs: Observations [B, T, C, H, W]
            hidden_state: Optional (h, c) hidden states

        Returns:
            policy_logits: [B, T, action_dim]
            values: [B, T, 1]
            hidden_state: (h, c)
        """
        B, T, C, H, W = obs.shape

        obs_flat = obs.view(B * T, C, H, W)

        h = obs_flat
        for conv_block in self.conv_blocks:
            h = conv_block(h)

        h = h.mean(dim=[2, 3])
        h = h.view(B, T, -1)

        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(h)
        else:
            lstm_out, hidden_state = self.lstm(h, hidden_state)

        policy_logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out)

        return policy_logits, values, hidden_state

    def get_action(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action from a single observation.

        Args:
            obs: Single observation [B, C, H, W]
            hidden_state: Optional (h, c) hidden states
            deterministic: If True, take argmax; else sample

        Returns:
            action: Selected action [B]
            hidden_state: (h, c)
        """
        obs = obs.unsqueeze(1)
        policy_logits, values, hidden_state = self.forward(obs, hidden_state)

        policy_logits = policy_logits.squeeze(1)

        if deterministic:
            action = policy_logits.argmax(dim=-1)
        else:
            action = torch.multinomial(F.softmax(policy_logits, dim=-1), 1).squeeze(-1)

        return int(action.item()), hidden_state

    def get_value(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get value for a single observation."""
        obs = obs.unsqueeze(1)
        _, values, hidden_state = self.forward(obs, hidden_state)
        return values.squeeze(1), hidden_state

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden states."""
        h = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return (h, c)

    def get_hidden_size(self) -> int:
        """Get LSTM hidden size."""
        return self.lstm.hidden_size


class RLLoss(nn.Module):
    """
    RL loss functions for DIAMOND.
    Implements REINFORCE with value baseline and λ-returns.
    """

    def __init__(
        self,
        discount_factor: float = 0.985,
        lambda_returns: float = 0.95,
        entropy_weight: float = 0.001,
    ):
        super().__init__()
        self.discount_factor = discount_factor
        self.lambda_returns = lambda_returns
        self.entropy_weight = entropy_weight

    def compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute λ-returns.

        Args:
            rewards: [B, T]
            values: [B, T+1]
            dones: [B, T]

        Returns:
            lambda_returns: [B, T]
        """
        B, T = rewards.shape

        rewards = torch.clamp(rewards, -10, 10)

        lambda_returns = torch.zeros_like(rewards)

        returns = values[:, -1]

        for t in reversed(range(T)):
            returns = rewards[:, t] + self.discount_factor * (
                1 - dones[:, t].float()
            ) * (
                (1 - self.lambda_returns) * values[:, t + 1]
                + self.lambda_returns * returns
            )
            lambda_returns[:, t] = returns

        return lambda_returns

    def policy_loss(
        self,
        policy_logits: torch.Tensor,
        actions: torch.Tensor,
        lambda_returns: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute policy loss with REINFORCE and entropy regularization.

        Args:
            policy_logits: [B, T, A]
            actions: [B, T]
            lambda_returns: [B, T]
            values: [B, T+1]

        Returns:
            policy_loss: scalar
        """
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        advantages = lambda_returns - values[:, :-1].detach()

        policy_loss = -(action_log_probs * advantages.detach()).mean()

        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_weight * entropy

        return policy_loss + entropy_loss

    def value_loss(
        self,
        values: torch.Tensor,
        lambda_returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value loss (MSE between value and lambda returns)."""
        target = lambda_returns.detach()
        value_pred = values[:, :-1].squeeze(-1)
        return F.mse_loss(value_pred, target)
