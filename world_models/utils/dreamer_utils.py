import os
import pickle
import torch
import numpy as np
import moviepy as mpy
import cv2

from typing import Iterable
from torch.nn import Module

from types import ModuleType
from typing import Optional

# Optional WandB import; keep typed for static checkers.
wandb: Optional[ModuleType] = None
try:
    import wandb as _wandb

    wandb = _wandb
except ImportError:
    wandb = None


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform used by Dreamer V2 for reward/value targets.

    Defined as ``sign(x) * log(1 + |x|)``. This compresses large positive or
    negative values into a range that is easier to predict with a categorical
    distribution over a bounded set of buckets.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`.

    Defined as ``sign(x) * (exp(|x|) - 1)``.
    """
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


class TwoHotEncoder:
    """Two-hot encoding for symlog targets (Dreamer V2 reward/value heads).

    A target value is softly assigned to the two nearest buckets on a uniform
    grid spanning ``[-symlog_range, symlog_range]``. The categorical logits
    produced by a network can then be decoded back into a real value by
    computing the expected bucket center.

    Args:
        num_buckets: Number of buckets in the categorical distribution.
        symlog_range: Maximum absolute value (in symlog space) covered by the
            grid. Values outside the range are clipped to the boundary buckets.
    """

    def __init__(self, num_buckets: int = 255, symlog_range: float = 10.0):
        self.num_buckets = int(num_buckets)
        self.symlog_range = float(symlog_range)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Allocate the bucket-center buffer on CPU. Use :meth:`to` to move."""
        self._bucket_centers = torch.linspace(
            -self.symlog_range, self.symlog_range, self.num_buckets
        )
        self.bucket_centers = self._bucket_centers

    def to(self, device: torch.device) -> "TwoHotEncoder":
        self.bucket_centers = self._bucket_centers.to(device)
        return self

    def encode(self, target: torch.Tensor) -> torch.Tensor:
        """Two-hot encode a real-valued target into soft bucket probabilities.

        Args:
            target: Tensor of arbitrary shape containing real-valued targets.

        Returns:
            Tensor with an extra final dimension of size ``num_buckets``
            containing the soft two-hot distribution. The encoding assumes
            the target is already in symlog space, matching Dreamer V2.
        """
        sym_target = torch.clamp(target, -self.symlog_range, self.symlog_range)
        target_flat = sym_target.reshape(-1)
        step = 2.0 * self.symlog_range / (self.num_buckets - 1)
        offset = (sym_target + self.symlog_range) / step
        offset_flat = offset.reshape(-1)
        lower = torch.floor(offset_flat).long()
        upper = torch.clamp(lower + 1, max=self.num_buckets - 1)
        lower = torch.clamp(lower, min=0)
        weight_upper = (offset_flat - lower.float()).unsqueeze(-1)
        weight_lower = 1.0 - weight_upper
        one_hot = torch.zeros(
            target_flat.shape[0], self.num_buckets, device=target_flat.device
        )
        one_hot.scatter_(1, lower.unsqueeze(-1), weight_lower)
        one_hot.scatter_add_(1, upper.unsqueeze(-1), weight_upper)
        return one_hot.reshape(*target.shape, self.num_buckets)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode categorical logits into the expected real-valued prediction.

        The logits are first softmaxed and then combined with the bucket
        centers. The output is passed through :func:`symexp` to invert the
        symlog transform.

        Args:
            logits: Tensor with a final dimension of ``num_buckets``.

        Returns:
            Tensor with the same shape as ``logits`` minus the last dimension.
        """
        probs = torch.softmax(logits, dim=-1)
        centers = self.bucket_centers.to(probs.device)
        expectation = (probs * centers).sum(-1, keepdim=True)
        return symexp(expectation)


def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    """Context manager that temporarily disables gradients for given modules.

    Useful during imagination or target-network forward passes where gradients
    through certain components should be blocked for speed and correctness.
    """

    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
          output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


class Logger:
    """Experiment logger for scalars and GIF rollouts using WandB.

    Provides helpers to write scalar metrics, dump pickle snapshots, and save
    video previews during Dreamer training/evaluation.
    """

    def __init__(
        self,
        log_dir,
        enable_wandb=False,
        wandb_api_key="",
        wandb_project="torchwm",
        wandb_entity="",
        video_format="gif",
        video_fps=20,
    ):
        self._log_dir = log_dir
        print("########################")
        print("logging outputs to ", log_dir)
        print("########################")
        self._n_logged_samples = 10
        self.enable_wandb = enable_wandb
        self.video_format = video_format
        self.video_fps = video_fps
        self._wandb_run = None

        if self.enable_wandb:
            if not wandb_api_key:
                raise ValueError("WandB API key is required when enable_wandb is True")
            if wandb is None:
                raise ImportError("wandb is not installed")
            os.environ["WANDB_API_KEY"] = wandb_api_key
            self._wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                dir=log_dir,
                name=os.path.basename(log_dir),
            )

    def log_scalar(self, scalar, name, step_):
        if self.enable_wandb and self._wandb_run:
            self._wandb_run.log({name: scalar}, step=step_)

    def log_scalars(self, scalar_dict, step):
        for key, value in scalar_dict.items():
            print("{} : {}".format(key, value))
            self.log_scalar(value, key, step)
        self.dump_scalars_to_pickle(scalar_dict, step)

    def log_videos(
        self, videos, step, max_videos_to_save=1, fps=None, video_title="video"
    ):
        if fps is None:
            fps = self.video_fps
        format = self.video_format
        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, videos.shape[0]])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0] > max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0] < max_length:
                padding = np.tile(
                    [videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1)
                )
                videos[i] = np.concatenate([videos[i], padding], 0)

            if format.lower() == "mp4":
                # Convert to uint8 HWC BGR for OpenCV
                video_u8 = (videos[i] * 255).astype(np.uint8)
                if video_u8.shape[-1] == 3:  # RGB to BGR
                    video_u8 = video_u8[..., ::-1]
                new_video_title = video_title + "{}_{}".format(step, i) + ".mp4"
                filename = os.path.join(self._log_dir, new_video_title)
                height, width = video_u8.shape[1], video_u8.shape[2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                for frame in video_u8:
                    out.write(frame)
                out.release()
            else:  # gif
                clip = mpy.ImageSequenceClip(list(videos[i]), fps=fps)
                new_video_title = video_title + "{}_{}".format(step, i) + ".gif"
                filename = os.path.join(self._log_dir, new_video_title)
                clip.write_gif(filename, fps=fps)

            # Log to WandB
            if self.enable_wandb and self._wandb_run:
                # Convert to numpy array for WandB
                video_array = np.array(videos[i])  # Shape: (T, H, W, C)
                # WandB expects (T, H, W, C) for Video
                self._wandb_run.log(
                    {f"{video_title}_{i}": wandb.Video(video_array, fps=fps)}, step=step
                )

    def dump_scalars_to_pickle(self, metrics, step, log_title=None):
        log_path = os.path.join(
            self._log_dir, "scalar_data.pkl" if log_title is None else log_title
        )
        with open(log_path, "ab") as f:
            pickle.dump({"step": step, **dict(metrics)}, f)

    def flush(self):
        pass


def compute_return(rewards, values, discounts, td_lam, last_value):
    """Compute TD(lambda) returns from imagined rewards, values, and discounts.

    Implements backward recursion used by Dreamer actor/value objectives.
    """
    next_values = torch.cat([values[1:], last_value.unsqueeze(0)], 0)
    targets = rewards + discounts * next_values * (1 - td_lam)
    rets = []
    last_rew = last_value

    for t in range(rewards.shape[0] - 1, -1, -1):
        last_rew = targets[t] + discounts[t] * td_lam * (last_rew)
        rets.append(last_rew)

    returns = torch.flip(torch.stack(rets), [0])
    return returns
