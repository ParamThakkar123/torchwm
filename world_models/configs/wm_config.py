"""Configuration classes for World Models training."""

from dataclasses import dataclass, field
from typing import Any, Dict


class WMVAEConfig:
    """Configuration class for Variational Autoencoder (VAE) training.

    This class manages all hyperparameters and settings for training a ConvVAE
    model on observation data. It provides validation and dictionary conversion
    utilities.

    Attributes:
        height: Height of input images (pixels).
        width: Width of input images (pixels).
        device: Device to train on ('cpu' or 'cuda').
        train_batch_size: Number of samples per training batch.
        num_epochs: Total number of training epochs.
        latent_size: Dimensionality of the VAE latent space.
        data_dir: Path to the dataset directory.
        learning_rate: Initial learning rate for optimizer.
        logdir: Directory for saving logs and checkpoints.
        noreload: If True, skip loading existing checkpoints.
        nosamples: If True, skip saving sample images during training.
        scheduler_patience: Epochs to wait before reducing learning rate.
        scheduler_factor: Multiplicative factor for learning rate reduction.
        early_stopping_patience: Epochs to wait before early stopping.
        sample_interval: Epoch interval for saving sample images.
        extra: Dictionary for additional custom parameters.

    Example:
        >>> config = WMVAEConfig({
        ...     'height': 64,
        ...     'width': 64,
        ...     'latent_size': 32,
        ...     'logdir': 'results',
        ... })
        >>> config.latent_size
        32
    """

    @dataclass
    class _Config:
        """Internal dataclass for VAE configuration."""

        height: int
        width: int
        device: str = "cpu"
        train_batch_size: int = 32
        num_epochs: int = 10
        latent_size: int = 32
        data_dir: str = "./data"
        learning_rate: float = 1e-3
        logdir: str = "results"
        noreload: bool = False
        nosamples: bool = False
        scheduler_patience: int = 5
        scheduler_factor: float = 0.5
        early_stopping_patience: int = 30
        sample_interval: int = 5
        extra: Dict[str, Any] = field(default_factory=dict)

        def validate(self) -> bool:
            """Validate configuration parameters.

            Raises:
                AssertionError: If any parameter is invalid.
            """
            assert self.height > 0 and self.width > 0, "height/width must be > 0"
            assert self.train_batch_size > 0, "train_batch_size must be > 0"
            assert self.num_epochs > 0, "num_epochs must be > 0"
            assert self.latent_size > 0, "latent_size must be > 0"
            assert self.learning_rate > 0.0, "learning_rate must be > 0"
            return True

        def to_dict(self) -> Dict[str, Any]:
            """Convert configuration to dictionary.

            Returns:
                Dictionary containing all configuration parameters.
            """
            d = {k: getattr(self, k) for k in self.__annotations__ if k != "extra"}
            d.update(self.extra)
            return d

    def __init__(self, config_dict: dict):
        """Initialize VAE configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.
        """
        known_keys = {
            "height",
            "width",
            "device",
            "train_batch_size",
            "num_epochs",
            "latent_size",
            "data_dir",
            "learning_rate",
            "logdir",
            "noreload",
            "nosamples",
            "scheduler_patience",
            "scheduler_factor",
            "early_stopping_patience",
            "sample_interval",
        }
        known = {k: v for k, v in config_dict.items() if k in known_keys}
        extra = {k: v for k, v in config_dict.items() if k not in known_keys}
        self._cfg = WMVAEConfig._Config(**known, extra=extra)
        self._cfg.validate()

    def __getattr__(self, item):
        """Get configuration attribute.

        Args:
            item: Name of the attribute to retrieve.

        Returns:
            Value of the configuration attribute.
        """
        return getattr(self._cfg, item)

    def to_dict(self):
        """Convert configuration to dictionary.

        Returns:
            Dictionary containing all configuration parameters.
        """
        return self._cfg.to_dict()


class WMMDNRNNConfig:
    """Configuration class for Mixture Density Recurrent Neural Network (MDRNN) training.

    This class manages all hyperparameters and settings for training an MDRNN
    model on sequence data. It provides validation and dictionary conversion
    utilities.

    Attributes:
        latent_size: Dimensionality of the latent space from VAE.
        action_size: Dimensionality of action space.
        hidden_size: Number of hidden units in RNN.
        gmm_components: Number of Gaussian mixture components.
        device: Device to train on ('cpu' or 'cuda').
        batch_size: Number of sequences per batch.
        seq_len: Length of each sequence.
        num_epochs: Total number of training epochs.
        data_dir: Path to the dataset directory.
        learning_rate: Initial learning rate for optimizer.
        logdir: Directory for saving logs and checkpoints.
        noreload: If True, skip loading existing checkpoints.
        include_reward: If True, include reward prediction in loss.
        scheduler_patience: Epochs to wait before reducing learning rate.
        scheduler_factor: Multiplicative factor for learning rate reduction.
        early_stopping_patience: Epochs to wait before early stopping.
        extra: Dictionary for additional custom parameters.

    Example:
        >>> config = WMMDNRNNConfig({
        ...     'latent_size': 32,
        ...     'action_size': 3,
        ...     'hidden_size': 256,
        ...     'gmm_components': 5,
        ... })
        >>> config.hidden_size
        256
    """

    @dataclass
    class _Config:
        """Internal dataclass for MDRNN configuration."""

        latent_size: int = 32
        action_size: int = 3
        hidden_size: int = 256
        gmm_components: int = 5
        device: str = "cpu"
        batch_size: int = 16
        seq_len: int = 32
        num_epochs: int = 30
        data_dir: str = "./data"
        learning_rate: float = 1e-3
        logdir: str = "results"
        noreload: bool = False
        include_reward: bool = True
        scheduler_patience: int = 5
        scheduler_factor: float = 0.5
        early_stopping_patience: int = 30
        extra: Dict[str, Any] = field(default_factory=dict)

        def validate(self) -> bool:
            """Validate configuration parameters.

            Raises:
                AssertionError: If any parameter is invalid.
            """
            assert self.latent_size > 0, "latent_size must be > 0"
            assert self.action_size > 0, "action_size must be > 0"
            assert self.hidden_size > 0, "hidden_size must be > 0"
            assert self.gmm_components > 0, "gmm_components must be > 0"
            assert self.batch_size > 0, "batch_size must be > 0"
            assert self.seq_len > 0, "seq_len must be > 0"
            assert self.num_epochs > 0, "num_epochs must be > 0"
            assert self.learning_rate > 0.0, "learning_rate must be > 0"
            return True

        def to_dict(self) -> Dict[str, Any]:
            """Convert configuration to dictionary.

            Returns:
                Dictionary containing all configuration parameters.
            """
            d = {k: getattr(self, k) for k in self.__annotations__ if k != "extra"}
            d.update(self.extra)
            return d

    def __init__(self, config_dict: dict):
        """Initialize MDRNN configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.
        """
        known_keys = {
            "latent_size",
            "action_size",
            "hidden_size",
            "gmm_components",
            "device",
            "batch_size",
            "seq_len",
            "num_epochs",
            "data_dir",
            "learning_rate",
            "logdir",
            "noreload",
            "include_reward",
            "scheduler_patience",
            "scheduler_factor",
            "early_stopping_patience",
        }
        known = {k: v for k, v in config_dict.items() if k in known_keys}
        extra = {k: v for k, v in config_dict.items() if k not in known_keys}
        self._cfg = WMMDNRNNConfig._Config(**known, extra=extra)
        self._cfg.validate()

    def __getattr__(self, item):
        """Get configuration attribute.

        Args:
            item: Name of the attribute to retrieve.

        Returns:
            Value of the configuration attribute.
        """
        return getattr(self._cfg, item)

    def to_dict(self):
        """Convert configuration to dictionary.

        Returns:
            Dictionary containing all configuration parameters.
        """
        return self._cfg.to_dict()


class WMControllerConfig:
    """Configuration class for Controller training with CMA-ES.

    This class manages hyperparameters for training a linear controller
    using Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    Attributes:
        latent_size: Dimensionality of latent state from VAE.
        hidden_size: Dimensionality of RSSM hidden state.
        action_size: Dimensionality of action space.
        logdir: Directory for saving logs and checkpoints.
        n_samples: Number of samples used to obtain return estimate.
        pop_size: Population size for CMA-ES.
        target_return: Stop once the return gets above this threshold.
        max_workers: Maximum number of workers for parallel evaluation.
        display: If True, show progress bars during training.
        time_limit: Maximum steps per episode.
    """

    @dataclass
    class _Config:
        """Internal dataclass for Controller configuration."""

        latent_size: int = 32
        hidden_size: int = 200
        action_size: int = 3
        logdir: str = "results"
        n_samples: int = 4
        pop_size: int = 10
        target_return: float = 950.0
        max_workers: int = 32
        display: bool = True
        time_limit: int = 1000
        extra: Dict[str, Any] = field(default_factory=dict)

        def validate(self) -> bool:
            """Validate configuration parameters."""
            assert self.latent_size > 0, "latent_size must be > 0"
            assert self.hidden_size > 0, "hidden_size must be > 0"
            assert self.action_size > 0, "action_size must be > 0"
            assert self.n_samples > 0, "n_samples must be > 0"
            assert self.pop_size > 0, "pop_size must be > 0"
            assert self.max_workers > 0, "max_workers must be > 0"
            return True

        def to_dict(self) -> Dict[str, Any]:
            """Convert configuration to dictionary."""
            d = {k: getattr(self, k) for k in self.__annotations__ if k != "extra"}
            d.update(self.extra)
            return d

    def __init__(self, config_dict: dict):
        """Initialize Controller configuration from dictionary."""
        known_keys = {
            "latent_size",
            "hidden_size",
            "action_size",
            "logdir",
            "n_samples",
            "pop_size",
            "target_return",
            "max_workers",
            "display",
            "time_limit",
        }
        known = {k: v for k, v in config_dict.items() if k in known_keys}
        extra = {k: v for k, v in config_dict.items() if k not in known_keys}
        self._cfg = WMControllerConfig._Config(**known, extra=extra)
        self._cfg.validate()

    def __getattr__(self, item):
        """Get configuration attribute."""
        return getattr(self._cfg, item)

    def to_dict(self):
        """Convert configuration to dictionary."""
        return self._cfg.to_dict()
