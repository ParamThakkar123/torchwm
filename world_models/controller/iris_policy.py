import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class IRISActor(nn.Module):
    """Actor network for IRIS policy.

    Takes reconstructed frames as input and outputs action logits.
    Uses CNN + LSTM architecture with burn-in mechanism.
    """

    def __init__(
        self,
        action_size: int,
        hidden_size: int = 512,
        num_layers: int = 4,
        frame_shape: Tuple[int, int, int] = (3, 64, 64),
    ):
        super().__init__()

        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.frame_shape = frame_shape

        # CNN feature extractor (shared with critic)
        self.cnn = CNNFeatureExtractor(frame_shape)

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=self.cnn.output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Action head
        self.action_head = nn.Linear(hidden_size, action_size)

    def forward(
        self,
        frames: torch.Tensor,  # (B, T, C, H, W) or (B, C, H, W)
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        burn_in_frames: Optional[torch.Tensor] = None,  # (B, burn_in, C, H, W)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through actor.

        Args:
            frames: Input frames (B, T, C, H, W) or (B, C, H, W)
            hidden_state: Optional (h, c) tuple for LSTM state
            burn_in_frames: Frames to use for initializing hidden state

        Returns:
            action_logits: Action logits (B, T, action_size) or (B, action_size)
            hidden_state: Updated (h, c) tuple
        """
        # Handle different input shapes
        if frames.dim() == 4:  # (B, C, H, W)
            frames = frames.unsqueeze(1)  # (B, 1, C, H, W)
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, C, H, W = frames.shape

        # Process each frame through CNN
        frames_flat = frames.reshape(B * T, C, H, W)
        features = self.cnn(frames_flat)  # (B*T, feature_size)
        features = features.reshape(B, T, -1)  # (B, T, feature_size)

        # Burn-in: initialize hidden state with past frames
        if burn_in_frames is not None:
            B_burn, T_burn, C_burn, H_burn, W_burn = burn_in_frames.shape
            burn_features = self.cnn(
                burn_in_frames.reshape(B_burn * T_burn, C_burn, H_burn, W_burn)
            )
            burn_features = burn_features.reshape(B_burn, T_burn, -1)

            # Initialize LSTM hidden state
            _, hidden_state = self.lstm(burn_features)

        # Process sequence through LSTM
        if hidden_state is None:
            hidden_state = self.init_hidden_state(B, frames.device)

        lstm_out, hidden_state = self.lstm(features, hidden_state)

        # Get action logits
        action_logits = self.action_head(lstm_out)  # (B, T, action_size)

        if squeeze_output:
            action_logits = action_logits.squeeze(1)  # (B, action_size)
            hidden_state = (hidden_state[0].squeeze(0), hidden_state[1].squeeze(0))

        return action_logits, hidden_state

    def init_hidden_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    def get_action(
        self,
        frame: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action from a single frame.

        Args:
            frame: Single frame (B, C, H, W)
            temperature: Softmax temperature (higher = more random)
            deterministic: If True, return argmax; else sample

        Returns:
            action: Selected action indices (B,)
        """
        self.eval()
        with torch.no_grad():
            action_logits, _ = self.forward(frame)
            action_logits = action_logits / temperature

            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)

        return action


class IRISCritic(nn.Module):
    """Critic network for IRIS value estimation.

    Shares CNN and LSTM with actor, but has separate value head.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 4,
        frame_shape: Tuple[int, int, int] = (3, 64, 64),
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.frame_shape = frame_shape

        # CNN feature extractor (shared with actor)
        self.cnn = CNNFeatureExtractor(frame_shape)

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=self.cnn.output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        frames: torch.Tensor,  # (B, T, C, H, W)
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through critic.

        Args:
            frames: Input frames (B, T, C, H, W)
            hidden_state: Optional (h, c) tuple

        Returns:
            values: Value estimates (B, T)
            hidden_state: Updated (h, c) tuple
        """
        B, T, C, H, W = frames.shape

        # CNN features
        frames_flat = frames.reshape(B * T, C, H, W)
        features = self.cnn(frames_flat)
        features = features.reshape(B, T, -1)

        # LSTM
        if hidden_state is None:
            hidden_state = self.init_hidden_state(B, frames.device)

        lstm_out, hidden_state = self.lstm(features, hidden_state)

        # Value
        values = self.value_head(lstm_out).squeeze(-1)  # (B, T)

        return values, hidden_state

    def init_hidden_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor shared between actor and critic.

    Processes input frames into feature vectors.
    """

    def __init__(
        self,
        frame_shape: Tuple[int, int, int] = (3, 64, 64),
        output_size: int = 512,
    ):
        super().__init__()

        self.frame_shape = frame_shape
        self.output_size = output_size

        # CNN layers: 64 -> 32 -> 16 -> 8 -> 4
        layers = []
        in_channels = frame_shape[0]

        channels = [32, 64, 128, 256]
        for i, out_channels in enumerate(channels):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Calculate output size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, *frame_shape)
            conv_out = self.conv(dummy)
            conv_size = conv_out.view(1, -1).shape[1]

        # Project to desired output size
        self.fc = nn.Linear(conv_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from frames.

        Args:
            x: Frames (B, C, H, W)

        Returns:
            features: Feature vectors (B, output_size)
        """
        B = x.shape[0]
        features = self.conv(x)
        features = features.reshape(B, -1)
        features = self.fc(features)
        return features


class IRISPolicy(nn.Module):
    """Combined policy module for IRIS.

    Wraps actor and optionally critic for convenience.
    """

    def __init__(
        self,
        action_size: int,
        hidden_size: int = 512,
        num_layers: int = 4,
        frame_shape: Tuple[int, int, int] = (3, 64, 64),
    ):
        super().__init__()

        self.actor = IRISActor(
            action_size=action_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            frame_shape=frame_shape,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Get action logits from frames."""
        action_logits, _ = self.actor(frames)
        return action_logits

    def act(
        self,
        frame: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample action from policy."""
        return self.actor.get_action(frame, temperature, deterministic)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state."""
        return self.actor.init_hidden_state(batch_size, device)
