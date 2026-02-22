"""Data generation and dataset classes for World Models.

This module provides utilities for generating rollout data from environments
and PyTorch dataset classes for loading observation sequences.
"""

import os
import gymnasium as gym
import numpy as np
from torch.utils.data import Dataset
from albumentations.core.composition import Compose
import glob
import torch
from bisect import bisect


def rollout(data):
    """Generate a rollout by running random actions in the environment.

    This function creates a single rollout by executing random actions in
    the CarRacing environment and saves the trajectory data to disk.

    Args:
        data: Tuple of (data_dir, seq_len, rollouts) containing:
            - data_dir: Directory to save rollout files
            - seq_len: Maximum length of each rollout
            - rollouts: Number of rollouts to generate

    Note:
        This function is designed to be used with multiprocessing.Pool
        for parallel data generation.
    """
    data_dir, seq_len, rollouts = data
    os.makedirs(data_dir)
    env = gym.make("CarRacing-v2", continuous=False)

    for i in range(rollouts):
        env.reset()
        actions_rollout = [env.action_space.sample() for _ in range(seq_len)]
        observations_rollout = []
        rewards_rollout = []
        dones_rollout = []

        t = 0
        while True:
            action = actions_rollout[t]
            t += 1

            obs, reward, done, truncated, _ = env.step(action)
            observations_rollout += [obs]
            rewards_rollout += [reward]
            dones_rollout += [done]

            if done or truncated:
                print(f"{data_dir.split('/')[-1]} | End of rollout {i} | {t} frames")
                np.savez(
                    os.path.join(data_dir, f"rollout_{i}"),
                    observations=np.array(observations_rollout),
                    rewards=np.array(rewards_rollout),
                    actions=np.array(actions_rollout),
                    terminals=np.array(dones_rollout),
                )
                break


class RolloutDataset(Dataset):
    """PyTorch Dataset for loading rollout data.

    This dataset loads pre-collected rollout trajectories from disk,
    providing a buffer-based mechanism for efficient data loading.
    It supports train/test splits and custom transforms.

    Attributes:
        root: Root directory containing rollout .npz files.
        transform: Albumentations transform to apply to observations.
        train: If True, use training split; otherwise use test split.
        buffer_size: Maximum number of files to keep in memory.
        num_test_files: Number of files to use for test set.

    Example:
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> dataset = RolloutDataset(
        ...     root='data/carracing',
        ...     transform=transform,
        ...     train=True,
        ...     buffer_size=100,
        ... )
        >>> obs, action, reward, terminal = dataset[0]
    """

    def __init__(
        self,
        root: str,
        transform: Compose,
        train: bool = True,
        buffer_size: int = 100,
        num_test_files: int = 600,
    ):
        """Initialize the RolloutDataset.

        Args:
            root: Root directory containing rollout .npz files.
            transform: Albumentations transform to apply to observations.
            train: If True, use training split; otherwise use test split.
            buffer_size: Maximum number of files to keep in memory.
            num_test_files: Number of files to use for test set.
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.files = glob.glob(self.root + "/**/*.npz", recursive=True)
        if train:
            self.files = self.files[:-num_test_files]
        else:
            self.files = self.files[-num_test_files:]

        self.cum_size = None
        self.buffer = None
        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.buffer_fnames = None

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples across all rollouts.
        """
        if not self.cum_size:
            print("Load new buffer")
            self.load_next_buffer()
        return self.cum_size[-1]

    def __getitem__(self, idx: int):
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (observation, action, reward, terminal) tensors.
        """
        file_idx = bisect(self.cum_size, idx)
        seq_idx = idx - self.cum_size[file_idx]
        data = self.buffer[file_idx]
        return self._get_data(data, seq_idx)

    def _get_data(self, data, idx):
        """Extract and transform a single data point from rollout.

        Args:
            data: Dictionary containing rollout data arrays.
            idx: Index within the rollout to extract.

        Returns:
            Tuple of (observation, action, reward, terminal) tensors.
        """
        obs = data["observations"][idx]
        action = data["actions"][idx]
        reward = data["rewards"][idx]
        terminal = data["terminals"][idx]

        if self.transform:
            obs = self.transform(image=obs)["image"]

        obs = torch.tensor(obs).permute(2, 0, 1).float() / 255.0
        action = torch.tensor(action).float()
        reward = torch.tensor(reward).float()
        terminal = torch.tensor(terminal).float()

        return dict(observation=obs, action=action, reward=reward, terminal=terminal)

    def load_next_buffer(self):
        """Load the next batch of rollout files into memory.

        This method implements a circular buffer, loading buffer_size files
        at a time and advancing through the dataset sequentially.
        """
        if self.buffer is None:
            self.buffer = [None] * len(self.files)
            self.cum_size = [0]
            self.buffer_fnames = [None] * len(self.files)

        start_idx = self.buffer_idx
        end_idx = min(start_idx + self.buffer_size, len(self.files))

        for i in range(start_idx, end_idx):
            if self.buffer[i] is None or self.buffer_fnames[i] != self.files[i]:
                self.buffer[i] = np.load(self.files[i])
                self.buffer_fnames[i] = self.files[i]

        self.buffer_idx = end_idx % len(self.files)

        sizes = [len(self.buffer[i]["observations"]) for i in range(start_idx, end_idx)]
        for s in sizes:
            self.cum_size.append(self.cum_size[-1] + s)


class ObservationDataset(RolloutDataset):
    """Dataset for single observation samples (not sequences).

    This dataset extends RolloutDataset to provide individual observations
    rather than sequences, suitable for VAE training.

    Example:
        >>> dataset = ObservationDataset(
        ...     root='data/carracing',
        ...     transform=transform,
        ...     train=True,
        ... )
        >>> obs = dataset[0]
    """

    def _get_data(self, data, idx: int):
        """Extract a single observation from rollout data.

        Args:
            data: Dictionary containing rollout data arrays.
            idx: Index of the observation to extract.

        Returns:
            torch.Tensor: Processed observation tensor.
        """
        obs = data["observations"][idx]
        if self.transform:
            transformed = self.transform(image=obs)
            obs = transformed["image"]
        obs = torch.tensor(obs).float()
        obs = obs / 255.0
        return obs


class SequenceDataset(RolloutDataset):
    """Dataset for sequential rollout data.

    This dataset provides sequences of observations, actions, rewards,
    and terminal flags suitable for training recurrent models like MDRNN.

    Attributes:
        seq_len: Length of sequences to return.

    Example:
        >>> dataset = SequenceDataset(
        ...     root='data/carracing',
        ...     transform=transform,
        ...     train=True,
        ...     seq_len=32,
        ... )
        >>> obs, action, reward, terminal, next_obs = dataset[0]
    """

    def __init__(
        self,
        root: str,
        transform: Compose,
        train: bool,
        buffer_size: int,
        num_test_files: int,
        seq_len: int,
    ):
        """Initialize the SequenceDataset.

        Args:
            root: Root directory containing rollout .npz files.
            transform: Albumentations transform to apply to observations.
            train: If True, use training split; otherwise use test split.
            buffer_size: Maximum number of files to keep in memory.
            num_test_files: Number of files to use for test set.
            seq_len: Length of sequences to return.
        """
        super().__init__(root, transform, train, buffer_size, num_test_files)
        self.seq_len = seq_len

    def _get_data(self, data, idx: int):
        """Extract a sequence of data from rollout.

        Args:
            data: Dictionary containing rollout data arrays.
            idx: Starting index of the sequence.

        Returns:
            Tuple of (observations, actions, rewards, terminals, next_observations).
        """
        obs_data = data["observations"][idx : idx + self.seq_len]
        if self.transform:
            transformed = [self.transform(image=obs) for obs in obs_data]
            obs_data = [t["image"] for t in transformed]

        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data["actions"][idx + 1 : idx + self.seq_len + 1]
        action = action.astype(np.float32)
        reward = data["rewards"][idx + 1 : idx + self.seq_len + 1]
        terminal = data["terminals"][idx + 1 : idx + self.seq_len + 1].astype(
            np.float32
        )
        return obs, action, reward, terminal, next_obs


class LatentSequenceDataset(Dataset):
    """Dataset for pre-computed latent sequences.

    This dataset uses pre-encoded latent representations instead of raw images,
    which significantly reduces memory usage during RNN training.
    """

    def __init__(
        self,
        latents: int,
        latents_arr: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        train: bool,
        buffer_size: int,
        num_test_files: int,
        seq_len: int,
    ):
        super().__init__()
        self.latents_arr = latents_arr
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        total_samples = len(actions)
        if train:
            self.start_idx = 0
            self.end_idx = total_samples - num_test_files * 100
        else:
            self.start_idx = total_samples - num_test_files * 100
            self.end_idx = total_samples

        self.seq_len = seq_len
        self.cum_size = list(range(self.start_idx, self.end_idx + 1, seq_len))

    def __len__(self):
        return len(self.cum_size) - 1

    def __getitem__(self, idx: int):
        start = self.cum_size[idx]
        latent_obs = self.latents_arr[start : start + self.seq_len]
        latent_next_obs = self.latents_arr[start + 1 : start + self.seq_len + 1]
        action = self.actions[start + 1 : start + self.seq_len + 1]
        reward = self.rewards[start + 1 : start + self.seq_len + 1]
        terminal = self.terminals[start + 1 : start + self.seq_len + 1]

        return (
            torch.tensor(latent_obs, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(terminal, dtype=torch.float32),
            torch.tensor(latent_next_obs, dtype=torch.float32),
        )
