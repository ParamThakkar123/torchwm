__all__ = [
    "VideoDatasetBase",
    "VideoFolderDataset",
    "ImageFolderDataset",
    "NumPyDataset",
    "RLEnvironmentDataset",
    "DatasetConfig",
    "VideoDatasetConfig",
    "create_video_dataloader",
    "create_video_dataset_from_config",
]


def __getattr__(name):
    if name == "VideoDatasetBase":
        from .video_datasets import VideoDatasetBase

        return VideoDatasetBase
    if name == "VideoFolderDataset":
        from .video_datasets import VideoFolderDataset

        return VideoFolderDataset
    if name == "ImageFolderDataset":
        from .video_datasets import ImageFolderDataset

        return ImageFolderDataset
    if name == "NumPyDataset":
        from .video_datasets import NumPyDataset

        return NumPyDataset
    if name == "RLEnvironmentDataset":
        from .video_datasets import RLEnvironmentDataset

        return RLEnvironmentDataset
    if name == "DatasetConfig":
        from .video_datasets import DatasetConfig

        return DatasetConfig
    if name == "VideoDatasetConfig":
        from .video_datasets import VideoDatasetConfig

        return VideoDatasetConfig
    if name == "create_video_dataloader":
        from .video_datasets import create_video_dataloader

        return create_video_dataloader
    if name == "create_video_dataset_from_config":
        from .video_datasets import create_video_dataset_from_config

        return create_video_dataset_from_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
