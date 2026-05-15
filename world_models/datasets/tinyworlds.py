"""
TinyWorlds Dataset Loaders

Loads pre-processed video datasets from HuggingFace for training Genie-style world models.
Based on: https://github.com/AlmondGod/tinyworlds

Available datasets:
- PICO_DOOM: Minimal Doom gameplay
- PONG: Classic Pong
- ZELDA: Zelda Ocarina of Time (2D)
- POLE_POSITION: Racing game
- SONIC: Sonic the Hedgehog
"""

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import hashlib
import urllib.request
import zipfile
import shutil

logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError:
    h5py = None
    logger.warning("h5py not installed. Install with: pip install h5py")

try:
    from huggingface_hub import hf_hub_download, list_repo_files

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning(
        "huggingface_hub not installed. Install with: pip install huggingface_hub"
    )


@dataclass
class TinyWorldsConfig:
    """Configuration for TinyWorlds datasets."""

    dataset_name: str = "SONIC"
    num_frames: int = 16
    image_size: int = 64
    batch_size: int = 4
    num_workers: int = 4
    cache_dir: Optional[str] = None
    split: str = "train"


class TinyWorldsDataset(Dataset):
    """Dataset for TinyWorlds game video data.

    Loads pre-processed frames from HuggingFace datasets repository.
    """

    DATASET_CONFIGS = {
        "PICO_DOOM": {
            "repo_id": "AlmondGod/tinyworlds",
            "filename": "picodoom_frames.h5",
            "description": "Minimal Doom gameplay",
        },
        "PONG": {
            "repo_id": "AlmondGod/tinyworlds",
            "filename": "pong_frames.h5",
            "description": "Classic Pong",
        },
        "ZELDA": {
            "repo_id": "AlmondGod/tinyworlds",
            "filename": "zelda_frames.h5",
            "description": "Zelda Ocarina of Time (2D)",
        },
        "POLE_POSITION": {
            "repo_id": "AlmondGod/tinyworlds",
            "filename": "pole_position_frames.h5",
            "description": "Racing game",
        },
        "SONIC": {
            "repo_id": "AlmondGod/tinyworlds",
            "filename": "sonic_frames.h5",
            "description": "Sonic the Hedgehog",
        },
    }

    def __init__(
        self,
        dataset_name: str = "SONIC",
        num_frames: int = 16,
        image_size: int = 64,
        split: str = "train",
        cache_dir: Optional[str] = None,
        download: bool = True,
        data_file: Optional[str] = None,
    ):
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(self.DATASET_CONFIGS.keys())}"
            )

        self.dataset_name = dataset_name
        self.config = self.DATASET_CONFIGS[dataset_name]
        self.num_frames = num_frames
        self.image_size = image_size
        self.split = split
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.data_file = data_file

        self._data_file = None
        self.num_samples = 0
        self.video_length = 0

        if data_file:
            self._load_data(Path(data_file))
        elif download:
            self._download_or_load_data()
        else:
            self._load_data()

    def _get_default_cache_dir(self) -> str:
        cache = os.path.expanduser("~/.cache/tinyworlds")
        os.makedirs(cache, exist_ok=True)
        return cache

    def _get_local_path(self) -> Path:
        return Path(self.cache_dir) / self.config["filename"]

    def _download_or_load_data(self):
        local_path = self._get_local_path()

        if local_path.exists():
            logger.info(f"Found cached dataset at {local_path}")
            self._load_data()
            return

        if not HF_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub not installed. Cannot download datasets.\n"
                "Install with: pip install huggingface_hub"
            )

        logger.info(f"Downloading {self.dataset_name} dataset from HuggingFace...")
        logger.info(f"This may take several minutes depending on your connection.")

        try:
            downloaded_path = hf_hub_download(
                repo_id=self.config["repo_id"],
                filename=self.config["filename"],
                repo_type="dataset",
                cache_dir=self.cache_dir,
            )
            downloaded_file = Path(downloaded_path)
            if not downloaded_file.exists():
                raise FileNotFoundError(f"Downloaded file not found: {downloaded_path}")

            local_path = self._get_local_path()
            if downloaded_file != local_path and not local_path.exists():
                import shutil

                shutil.copy2(downloaded_file, local_path)

            logger.info(f"Downloaded to: {local_path}")
            self._load_data()
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {self.dataset_name}: {e}\n"
                "You can manually download from: "
                f"https://huggingface.co/datasets/{self.config['repo_id']}/tree/main"
            )

    def _load_data(self, file_path: Optional[Path] = None):
        local_path = file_path or self._get_local_path()

        if not local_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {local_path}")

        self._data_file = h5py.File(local_path, "r")

        if "videos" in self._data_file:
            data = self._data_file["videos"]
        elif "frames" in self._data_file:
            data = self._data_file["frames"]
        else:
            available_keys = list(self._data_file.keys())
            raise KeyError(
                f"No 'videos' or 'frames' key found. Available: {available_keys}"
            )

        if len(data.shape) == 5:
            self.num_samples = data.shape[0]
            self.video_length = data.shape[1]
            self.raw_height = data.shape[2]
            self.raw_width = data.shape[3]
            self.channels = data.shape[4]
            self.data_layout = "NCTHW" if data.shape[-1] <= 4 else "NTHWC"
        elif len(data.shape) == 4:
            self.num_samples = data.shape[0]
            self.video_length = data.shape[1]
            self.raw_height = data.shape[2]
            self.raw_width = data.shape[3]
            self.channels = 1
            self.data_layout = "NTHW"
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        logger.info(
            f"Loaded {self.dataset_name}: {self.num_samples} videos, "
            f"{self.video_length} frames each, {self.raw_height}x{self.raw_width}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = self._data_file["videos" if "videos" in self._data_file else "frames"]
        video = data[idx][:]

        if not isinstance(video, np.ndarray):
            video = np.array(video)

        if self.data_layout == "NCTHW":
            video = np.transpose(video, (0, 2, 3, 1))
        elif self.data_layout == "NTHWC":
            video = np.transpose(video, (0, 1, 3, 2))
        elif self.data_layout == "NTHW":
            pass
        else:
            raise ValueError(f"Unknown data layout: {self.data_layout}")

        if len(video.shape) == 3:
            if video.shape[-1] == 1:
                video = np.repeat(video, 3, axis=-1)
            else:
                video = np.expand_dims(video, axis=-1)
                video = np.repeat(video, 3, axis=-1)

        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            video = video[indices]
        else:
            padding = np.tile(video[-1:], (self.num_frames - total_frames, 1, 1, 1))
            video = np.concatenate([video, padding], axis=0)

        processed = []
        for frame in video:
            if frame.max() <= 1.0 and frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame)
            img = img.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            processed.append(np.array(img))
        video = np.stack(processed)

        if video.max() <= 1.0:
            video = video / 255.0

        video = torch.from_numpy(video).float()
        video = video.permute(0, 3, 1, 2)

        return video

    def __del__(self):
        if self._data_file is not None:
            self._data_file.close()

    def get_info(self) -> Dict:
        """Return dataset information."""
        return {
            "name": self.dataset_name,
            "description": self.config["description"],
            "num_samples": self.num_samples,
            "video_length": self.video_length,
            "raw_resolution": f"{self.raw_height}x{self.raw_width}",
            "channels": self.channels,
        }


class TinyWorldsDataLoader:
    """Factory class for creating TinyWorlds dataloaders."""

    DATASET_NAMES = list(TinyWorldsDataset.DATASET_CONFIGS.keys())

    @staticmethod
    def create_dataloader(
        dataset_name: str = "SONIC",
        num_frames: int = 16,
        image_size: int = 64,
        batch_size: int = 4,
        num_workers: int = 4,
        shuffle: bool = True,
        cache_dir: Optional[str] = None,
        download: bool = True,
        data_file: Optional[str] = None,
    ) -> Tuple[TinyWorldsDataset, DataLoader]:
        """Create a dataloader for TinyWorlds dataset.

        Args:
            dataset_name: Name of the game dataset (PICO_DOOM, PONG, ZELDA, POLE_POSITION, SONIC)
            num_frames: Number of frames per video sequence
            image_size: Target image size (will resize frames)
            batch_size: Batch size
            num_workers: Number of data loading workers
            shuffle: Whether to shuffle the data
            cache_dir: Directory to cache downloaded datasets
            download: Whether to download if not cached

        Returns:
            Tuple of (dataset, dataloader)

        Usage:
            dataset, loader = TinyWorldsDataLoader.create_dataloader(
                dataset_name="SONIC",
                num_frames=16,
                image_size=64,
                batch_size=4
            )
        """
        dataset = TinyWorldsDataset(
            dataset_name=dataset_name,
            num_frames=num_frames,
            image_size=image_size,
            cache_dir=cache_dir,
            download=download,
            data_file=data_file,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle,
        )

        logger.info(
            f"Created {dataset_name} dataloader: {len(dataset)} samples, "
            f"{len(loader)} batches, batch_size={batch_size}"
        )

        return dataset, loader

    @staticmethod
    def list_available_datasets() -> List[str]:
        """List all available dataset names."""
        return TinyWorldsDataLoader.DATASET_NAMES

    @staticmethod
    def get_dataset_info(dataset_name: str) -> Dict:
        """Get information about a specific dataset without downloading."""
        if dataset_name not in TinyWorldsDataset.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(TinyWorldsDataset.DATASET_CONFIGS.keys())}"
            )
        return TinyWorldsDataset.DATASET_CONFIGS[dataset_name]


def create_tinyworlds_dataloader(
    dataset_name: str = "SONIC",
    num_frames: int = 16,
    image_size: int = 64,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    cache_dir: Optional[str] = None,
    download: bool = True,
    data_file: Optional[str] = None,
) -> Tuple[TinyWorldsDataset, DataLoader]:
    """Factory function to create TinyWorlds dataloaders.

    Args:
        dataset_name: Name of the game dataset (PICO_DOOM, PONG, ZELDA, POLE_POSITION, SONIC)
        num_frames: Number of frames per video sequence
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle
        cache_dir: Cache directory for datasets
        download: Download if not cached

    Returns:
        Tuple of (dataset, dataloader)

    Usage:
        dataset, loader = create_tinyworlds_dataloader(
            dataset_name="SONIC",
            num_frames=16,
            batch_size=4
        )

        for batch in loader:
            # batch shape: (B, T, C, H, W)
            ...
    """
    return TinyWorldsDataLoader.create_dataloader(
        dataset_name=dataset_name,
        num_frames=num_frames,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        cache_dir=cache_dir,
        download=download,
        data_file=data_file,
    )


def download_all_datasets(cache_dir: Optional[str] = None) -> Dict[str, str]:
    """Download all available TinyWorlds datasets.

    Args:
        cache_dir: Directory to cache downloaded datasets

    Returns:
        Dictionary mapping dataset names to local file paths
    """
    cache_dir = cache_dir or os.path.expanduser("~/.cache/tinyworlds")
    results = {}

    for dataset_name in TinyWorldsDataLoader.list_available_datasets():
        logger.info(f"Downloading {dataset_name}...")
        try:
            dataset, _ = create_tinyworlds_dataloader(
                dataset_name=dataset_name,
                download=True,
                cache_dir=cache_dir,
            )
            results[dataset_name] = str(dataset._get_local_path())
            logger.info(f"  -> {dataset_name} downloaded successfully")
        except Exception as e:
            logger.error(f"  -> {dataset_name} failed: {e}")
            results[dataset_name] = None

    return results
