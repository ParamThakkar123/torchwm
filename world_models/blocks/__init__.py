"""
Blocks sub-module - Transformer blocks and attention mechanisms.

Exported Components:
    Transformers:
        - STTransformer: Spatiotemporal Transformer for video processing
        - STSpatialAttention: Spatial attention layer
        - STTemporalAttention: Temporal attention layer
        - STBlock: Combined spatiotemporal transformer block

    Attention:
        - MultiHeadSelfAttention: Multi-head self-attention
        - MultiHeadAttention: Generic multi-head attention (alias)
        - Attention: Attention mechanism

    Normalization:
        - RMSNorm: Root Mean Square Layer Normalization
        - AdaLNNormalization: Adaptive Layer Normalization
"""

__all__ = [
    # Transformers
    "STTransformer",
    "STSpatialAttention",
    "MultiHeadSelfAttention",
    "MultiHeadAttention",
    # Normalization
    "RMSNorm",
    "AdaLNNormalization",
]


def __getattr__(name):
    # Transformers
    if name == "STTransformer":
        from .st_transformer import STTransformer

        return STTransformer
    if name == "STSpatialAttention":
        from .st_transformer import STSpatialAttention

        return STSpatialAttention

    # Attention
    if name == "MultiHeadSelfAttention":
        from .mhsa import MultiHeadSelfAttention

        return MultiHeadSelfAttention
    if name == "MultiHeadAttention":
        from .mhsa import MultiHeadSelfAttention

        return MultiHeadSelfAttention  # Alias

    # Normalization
    if name == "RMSNorm":
        from world_models.layers.rms_norm import RMSNorm

        return RMSNorm
    if name == "AdaLNNormalization":
        from world_models.layers.ada_ln_norm import AdaLNNormalization

        return AdaLNNormalization

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
