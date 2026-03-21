import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization that conditions on actions and diffusion time."""

    def __init__(self, num_groups: int, num_channels: int, cond_dim: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.linear = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            cond: Conditioning tensor [B, cond_dim]
        """
        x = self.norm(x)
        scale, bias = self.linear(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + bias


class ResBlock(nn.Module):
    """Residual block with adaptive group normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(32, in_channels, cond_dim)
        self.norm2 = AdaptiveGroupNorm(32, out_channels, cond_dim)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, cond))
        h = self.conv1(h)
        h = self.act(self.norm2(h, cond))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for U-Net."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.channels = channels
        self.norm = AdaptiveGroupNorm(32, channels, cond_dim)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x, cond)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)

        h = torch.matmul(attn, v)
        h = self.proj(h)

        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, freq_dim),
            nn.SiLU(),
            nn.Linear(freq_dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        return self.mlp(t)


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        num_res_blocks: int = 2,
        attention: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(in_channels, out_channels, cond_dim))
            in_channels = out_channels

        if attention:
            self.attn = AttentionBlock(out_channels, cond_dim)
        else:
            self.attn = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x, cond)
        if self.attn is not None:
            x = self.attn(x, cond)
        return x


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        num_res_blocks: int = 2,
        attention: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(in_channels, out_channels, cond_dim))
            in_channels = out_channels

        if attention:
            self.attn = AttentionBlock(out_channels, cond_dim)
        else:
            self.attn = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x, cond)
        if self.attn is not None:
            x = self.attn(x, cond)
        return x


class DiffusionUNet(nn.Module):
    """
    U-Net architecture for EDM diffusion world model.
    Uses frame stacking for observation conditioning and adaptive group norm for action conditioning.
    """

    def __init__(
        self,
        obs_channels: int = 3,
        num_conditioning_frames: int = 4,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 1, 1, 1),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        action_dim: int = 18,
    ):
        super().__init__()
        self.num_conditioning_frames = num_conditioning_frames
        self.obs_channels = obs_channels

        self.input_conv = nn.Conv2d(
            obs_channels * (num_conditioning_frames + 1), base_channels, 3, padding=1
        )

        self.time_embed = TimestepEmbedding(cond_dim)

        self.action_embed = nn.Embedding(action_dim, cond_dim)

        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            self.down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    cond_dim,
                    num_res_blocks,
                    attention=False,
                )
            )
            in_ch = out_ch

        self.middle_block = nn.ModuleList(
            [
                ResBlock(in_ch, in_ch, cond_dim),
                AttentionBlock(in_ch, cond_dim),
                ResBlock(in_ch, in_ch, cond_dim),
            ]
        )

        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            self.up_blocks.append(
                UpBlock(
                    in_ch,
                    out_ch,
                    cond_dim,
                    num_res_blocks,
                    attention=False,
                )
            )
            in_ch = out_ch

        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, obs_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        obs_history: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Args:
            x: Noised observation at timestep t [B, C, H, W]
            t: Diffusion timestep [B]
            obs_history: Past observations for conditioning [B, L, C, H, W]
            actions: Past actions [B, L]

        Returns:
            Predicted clean observation [B, C, H, W]
        """
        B = x.shape[0]

        obs_history = obs_history.view(B, -1, *obs_history.shape[-2:])
        x = torch.cat([x, obs_history], dim=1)

        h = self.input_conv(x)

        t_emb = self.time_embed(t)
        action_emb = self.action_embed(actions.long())
        cond = t_emb + action_emb.sum(dim=1)

        skip_connections = []
        for down_block in self.down_blocks:
            h = down_block(h, cond)
            skip_connections.append(h)
            h = F.avg_pool2d(h, 2)

        for block in self.middle_block:
            if isinstance(block, AttentionBlock):
                h = block(h, cond)
            else:
                h = block(h, cond)

        for up_block in self.up_blocks:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = up_block(h, cond)

        return self.output_conv(h)


class EDMPreconditioner:
    """EDM preconditioner following Karras et al. (2022)."""

    def __init__(
        self,
        sigma_data: float = 0.5,
        p_mean: float = -0.4,
        p_std: float = 1.2,
    ):
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std

    def get_preconditioners(self, sigma: torch.Tensor) -> dict:
        """
        Compute EDM preconditioners for given noise levels.

        Returns:
            Dictionary with c_skip, c_out, c_in, c_noise
        """
        c_skip = (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = torch.log(sigma) / 4.0

        return {
            "c_skip": c_skip,
            "c_out": c_out,
            "c_in": c_in,
            "c_noise": c_noise,
        }

    def sample_noise_level(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise level from log-normal distribution."""
        log_sigma = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        sigma = torch.exp(log_sigma)
        return sigma

    def denoise(
        self, model, x: torch.Tensor, sigma: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Apply EDM denoising with preconditioners.

        Args:
            model: Diffusion model
            x: Noised input [B, C, H, W]
            sigma: Noise level [B]
            **kwargs: Additional conditioning (obs_history, actions)

        Returns:
            Denoised prediction [B, C, H, W]
        """
        sigma = sigma.view(-1, 1, 1, 1)
        precond = self.get_preconditioners(sigma)

        model_input = precond["c_in"] * x
        model_output = model(model_input, sigma.squeeze(-1).squeeze(-1), **kwargs)

        denoised = precond["c_skip"] * x + precond["c_out"] * model_output
        return denoised


class EulerSampler:
    """Euler method sampler for reverse diffusion."""

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: int = 7,
        num_steps: int = 3,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_steps = num_steps

        self.step_indices = torch.arange(num_steps)
        t_steps = (
            sigma_max ** (1 / rho)
            + self.step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        self.t_steps = torch.flip(t_steps, dims=(0,))
        self.t_next = torch.cat([self.t_steps[1:], torch.tensor([0.0])])

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        obs_history: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples using Euler method.

        Args:
            model: Diffusion model
            shape: Output shape [B, C, H, W]
            device: Device to run on
            obs_history: Conditioning observations [B, L, C, H, W]
            actions: Conditioning actions [B, L]

        Returns:
            Generated samples [B, C, H, W]
        """
        B = shape[0]
        x = torch.randn(shape, device=device) * self.t_steps[0]

        for i in range(self.num_steps):
            t_cur = self.t_steps[i].expand(B)
            t_next = self.t_next[i].expand(B)

            sigma_cur = t_cur.view(-1, 1, 1, 1)

            model_output = model(
                x,
                t_cur,
                obs_history=obs_history,
                actions=actions,
            )

            denoised = x + model_output * sigma_cur

            d_cur = (denoised - x) / sigma_cur
            x = denoised + (t_next.view(-1) - t_cur.view(-1)).view(-1, 1, 1, 1) * d_cur

        return x.clamp(0, 1)
