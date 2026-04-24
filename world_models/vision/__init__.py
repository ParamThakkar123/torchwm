__all__ = [
    "VideoTokenizer",
    "create_video_tokenizer",
    "VectorQuantizer",
    "VectorQuantizerEMA",
]


def __getattr__(name):
    if name == "VideoTokenizer":
        from .video_tokenizer import VideoTokenizer

        return VideoTokenizer
    if name == "create_video_tokenizer":
        from .video_tokenizer import create_video_tokenizer

        return create_video_tokenizer
    if name == "VectorQuantizer":
        from .vq_layer import VectorQuantizer

        return VectorQuantizer
    if name == "VectorQuantizerEMA":
        from .vq_layer import VectorQuantizerEMA

        return VectorQuantizerEMA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
