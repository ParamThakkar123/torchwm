import os
import pickle
import torch
import numpy as np
import moviepy as mpy
import cv2

from typing import Iterable
from torch.nn import Module
from tensorboardX import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None


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
    """Experiment logger for scalars and GIF rollouts using TensorBoardX and WandB.

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
        enable_tensorboard=True,
        video_format="gif",
        video_fps=20,
    ):
        self._log_dir = log_dir
        print("########################")
        print("logging outputs to ", log_dir)
        print("########################")
        self._n_logged_samples = 10
        self.enable_wandb = enable_wandb
        self.enable_tensorboard = enable_tensorboard
        self.video_format = video_format
        self.video_fps = video_fps
        self._wandb_run = None

        if self.enable_tensorboard:
            self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)
        else:
            self._summ_writer = None

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
        if self.enable_tensorboard and self._summ_writer:
            self._summ_writer.add_scalar("{}".format(name), scalar, step_)
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
        if self.enable_tensorboard and self._summ_writer:
            self._summ_writer.flush()


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
