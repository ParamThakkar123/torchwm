import os
import random
import time
from typing import Optional, Any, Tuple, List, Dict
import numpy as np
import torch
from collections import OrderedDict

from world_models.models.dreamer import Dreamer, make_env
from world_models.utils.dreamer_utils import Logger
from world_models.configs.dreamer_config import DreamerConfig


class DreamerAgent:
    """High-level user API for running Dreamer experiments end to end.

    It builds environments from config, initializes seeds and logging,
    instantiates `Dreamer`, and exposes simple `train()` / `evaluate()` methods.
    """

    def __init__(self, config: Optional[DreamerConfig] = None, **kwargs: Any) -> None:
        if config is None:
            self.args = DreamerConfig()
        else:
            self.args = config

        self.last_latents_ref: Optional[List[Optional[torch.Tensor]]] = kwargs.get(
            "last_latents_ref", None
        )

        for key, value in kwargs.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
            elif key == "logdir":
                # Accept either `logdir` (server/legacy) and mirror into
                # both `logdir` and `log_dir` so downstream code using either
                # naming convention picks up the value.
                setattr(self.args, "logdir", value)
                try:
                    setattr(self.args, "log_dir", value)
                except Exception:
                    pass
            elif key == "last_latents_ref":
                self.last_latents_ref = value
            else:
                raise ValueError(f"Invalid argument: {key}")

        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # Allow caller to pass an absolute `logdir`. If provided and absolute,
        # use it verbatim. Otherwise keep the historical behavior of creating
        # a subdir under the package data path so examples/tests remain stable.
        if hasattr(self.args, "logdir") and self.args.logdir is not None:
            self.logdir = self.args.logdir
        else:
            self.logdir = (
                self.args.env
                + "_"
                + self.args.algo
                + "_"
                + self.args.exp_name
                + "_"
                + time.strftime("%d-%m-%Y-%H-%M-%S")
            )

        # If `self.logdir` is not an absolute path, place it under package data_path
        if not os.path.isabs(self.logdir):
            self.logdir = os.path.join(data_path, self.logdir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available() and not self.args.no_gpu:
            device = torch.device("cuda")
            torch.cuda.manual_seed(self.args.seed)
        else:
            device = torch.device("cpu")

        self.train_env = make_env(self.args)
        self.test_env = make_env(self.args)

        obs_shape = self.train_env.observation_space["image"].shape
        action_size = self.train_env.action_space.shape[0]
        self.dreamer = Dreamer(
            self.args, obs_shape, action_size, device, self.args.restore
        )

        self.logger = Logger(
            self.logdir,
            self.args.enable_wandb,
            self.args.wandb_api_key,
            self.args.wandb_project,
            self.args.wandb_entity,
            self.args.video_format,
            self.args.video_fps,
        )

    def train(self, total_steps: Optional[int] = None) -> None:
        if total_steps is None:
            total_steps = self.args.total_steps

        initial_logs = OrderedDict()
        seed_episode_rews = self.dreamer.collect_random_episodes(
            self.train_env, self.args.seed_steps // self.args.action_repeat
        )
        global_step = self.dreamer.data_buffer.steps * self.args.action_repeat
        # without loss of generality initial rews for both train and eval are assumed same
        initial_logs.update(
            {
                "train_avg_reward": np.mean(seed_episode_rews),
                "train_max_reward": np.max(seed_episode_rews),
                "train_min_reward": np.min(seed_episode_rews),
                "train_std_reward": np.std(seed_episode_rews),
                "eval_avg_reward": np.mean(seed_episode_rews),
                "eval_max_reward": np.max(seed_episode_rews),
                "eval_min_reward": np.min(seed_episode_rews),
                "eval_std_reward": np.std(seed_episode_rews),
            }
        )
        self.logger.log_scalars(initial_logs, step=0)
        self.logger.flush()

        while global_step <= total_steps:
            print("##################################")
            print(f"At global step {global_step}")

            logs = OrderedDict()

            for _ in range(self.args.update_steps):
                model_loss, actor_loss, value_loss = self.dreamer.train_one_batch()

            train_rews = self.dreamer.act_and_collect_data(
                self.train_env, self.args.collect_steps // self.args.action_repeat
            )

            logs.update(
                {
                    "model_loss": model_loss,
                    "actor_loss": actor_loss,
                    "value_loss": value_loss,
                    "train_avg_reward": np.mean(train_rews),
                    "train_max_reward": np.max(train_rews),
                    "train_min_reward": np.min(train_rews),
                    "train_std_reward": np.std(train_rews),
                }
            )

            if global_step % self.args.test_interval == 0:
                episode_rews, video_images, latents = self.dreamer.evaluate(
                    self.test_env, self.args.test_episodes
                )
                if self.last_latents_ref is not None and latents is not None:
                    self.last_latents_ref[0] = latents

                logs.update(
                    {
                        "eval_avg_reward": np.mean(episode_rews),
                        "eval_max_reward": np.max(episode_rews),
                        "eval_min_reward": np.min(episode_rews),
                        "eval_std_reward": np.std(episode_rews),
                    }
                )

            self.logger.log_scalars(logs, global_step)

            if (
                global_step % self.args.log_video_freq == 0
                and self.args.log_video_freq != -1
                and len(video_images[0]) != 0
            ):
                self.logger.log_video(
                    video_images, global_step, self.args.max_videos_to_save
                )
            if global_step % self.args.checkpoint_interval == 0:
                ckpt_dir = os.path.join(self.logdir, "ckpts/")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                self.dreamer.save(os.path.join(ckpt_dir, f"{global_step}_ckpt.pt"))

            global_step = self.dreamer.data_buffer.steps * self.args.action_repeat
            self.logger.flush()

    def evaluate(self) -> Tuple[np.ndarray, List[np.ndarray], Optional[torch.Tensor]]:
        logs = OrderedDict()
        episode_rews, video_images, latents = self.dreamer.evaluate(
            self.test_env, self.args.test_episodes, render=True
        )
        if self.last_latents_ref is not None and latents is not None:
            self.last_latents_ref[0] = latents
        logs.update(
            {
                "test_avg_reward": np.mean(episode_rews),
                "test_max_reward": np.max(episode_rews),
                "test_min_reward": np.min(episode_rews),
                "test_std_reward": np.std(episode_rews),
            }
        )
        self.logger.dump_scalars_to_pickle(logs, 0, log_title="test_scalars.pkl")
        self.logger.log_videos(
            video_images, 0, max_videos_to_save=self.args.max_videos_to_save
        )
        self.logger.flush()
        return episode_rews, video_images, latents
