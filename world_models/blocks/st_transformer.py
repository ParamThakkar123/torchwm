import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class STSpatialAttention(nn.Module):
    """Spatial attention layer - attends over H*W tokens within each time step."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) where T is temporal dim, N is spatial dim (H*W)
        Returns:
            (B, T, N, C)
        """
        B, T, N, C = x.shape

        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, heads, T, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class STTemporalAttention(nn.Module):
    """Temporal attention layer - attends over T tokens across all spatial positions with causal mask."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) where T is temporal dim, N is spatial dim (H*W)
            causal: whether to apply causal masking
        Returns:
            (B, T, N, C)
        """
        B, T, N, C = x.shape

        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 2, 1, 5)  # (3, B, heads, N, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if causal:
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 3).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class STMLP(nn.Module):
    """MLP for ST-Transformer block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STTransformerBlock(nn.Module):
    """Spatiotemporal Transformer Block.

    Contains interleaved spatial and temporal attention layers, followed by a single MLP.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1_spatial = norm_layer(dim)
        self.norm1_temporal = norm_layer(dim)

        self.attn_spatial = STSpatialAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.attn_temporal = STTemporalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = STMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) or (B, T*H*W, C)
        Returns:
            Same shape as input
        """
        # Handle both (B, T*N, C) and (B, T, N, C) inputs
        if x.dim() == 3:
            B, T_N, C = x.shape
            # Infer T and N - we assume T=16 and N=H*W
            T = 16  # Default sequence length from paper
            N = T_N // T
            x = x.reshape(B, T, N, C)

        B, T, N, C = x.shape

        # Spatial attention (within each time step)
        x = x + self.drop_path(self.attn_spatial(self.norm1_spatial(x)))

        # Temporal attention (across time steps, with causal mask)
        x = x + self.drop_path(self.attn_temporal(self.norm1_temporal(x)))

        # MLP (single FFW after both spatial and temporal, as per paper)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class STTransformer(nn.Module):
    """Spatiotemporal Transformer for video modeling.

    Contains L spatiotemporal blocks with interleaved spatial and temporal attention.
    """

    def __init__(
        self,
        num_frames: int = 16,
        num_patches_per_frame: int = 256,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches_per_frame = num_patches_per_frame

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                STTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T*N, C) where T is num_frames, N is num_patches_per_frame
        Returns:
            (B, T*N, C)
        """
        B, T_N, C = x.shape
        T = self.num_frames
        N = T_N // T

        # Reshape to (B, T, N, C) for ST-attention
        x = x.reshape(B, T, N, C)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Reshape back to (B, T*N, C)
        x = x.reshape(B, T * N, C)

        return x


def create_st_transformer(
    num_frames: int = 16,
    patch_size: int = 4,
    img_size: int = 64,
    dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> STTransformer:
    """Factory function to create an ST-Transformer."""
    num_patches_per_frame = (img_size // patch_size) ** 2

    return STTransformer(
        num_frames=num_frames,
        num_patches_per_frame=num_patches_per_frame,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )
