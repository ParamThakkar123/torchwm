import os
import random
import time
from typing import Tuple, Any, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

from collections import OrderedDict

import world_models.envs.wrappers as env_wrapper
from world_models.envs.dmc import DeepMindControlEnv
from world_models.envs.gym_env import GymImageEnv
from world_models.envs.unity_env import UnityMLAgentsEnv
from world_models.memory.dreamer_memory import ReplayBuffer
from world_models.models.dreamer_rssm import RSSM
from world_models.vision.dreamer_decoder import ConvDecoder, DenseDecoder, ActionDecoder
from world_models.vision.dreamer_encoder import ConvEncoder
from world_models.utils.dreamer_utils import Logger, FreezeParameters, compute_return
from world_models.configs.dreamer_config import DreamerConfig

# Only set MUJOCO_GL for non-Windows platforms. On Windows the 'egl' value
# causes mujoco to raise a RuntimeError during import. Respect an existing
# environment value if present.
if os.name != "nt" and os.environ.get("MUJOCO_GL") is None:
    os.environ["MUJOCO_GL"] = "egl"


def _resolve_image_size(args: Any) -> Tuple[int, int]:
    size = getattr(args, "image_size", (64, 64))
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Invalid image_size={size}. Expected int or (H, W).")


def make_env(args: Any) -> Any:
    """Construct a Dreamer-compatible environment from `DreamerConfig` options.

    Supports DMC, Gym/Gymnasium, and Unity ML-Agents backends and applies the
    standard wrapper stack: action repeat, action normalization, and time limit.
    """
    size = _resolve_image_size(args)
    backend = str(getattr(args, "env_backend", "dmc")).lower()

    env_instance = getattr(args, "env_instance", None)
    if env_instance is not None:
        env = GymImageEnv(
            env_instance,
            seed=args.seed,
            size=size,
            render_mode=getattr(args, "gym_render_mode", "rgb_array"),
        )
    elif backend == "dmc":
        env = DeepMindControlEnv(args.env, args.seed, size=size)
    elif backend in {"gym", "gymnasium", "generic"}:
        env = GymImageEnv(
            args.env,
            seed=args.seed,
            size=size,
            render_mode=getattr(args, "gym_render_mode", "rgb_array"),
        )
    elif backend in {"unity", "unity_mlagents", "mlagents"}:
        unity_file_name = getattr(args, "unity_file_name", None)
        if not unity_file_name:
            raise ValueError(
                "unity_file_name must be provided when env_backend='unity_mlagents'."
            )
        env = UnityMLAgentsEnv(
            file_name=unity_file_name,
            behavior_name=getattr(args, "unity_behavior_name", None),
            seed=args.seed,
            size=size,
            worker_id=int(getattr(args, "unity_worker_id", 0)),
            base_port=int(getattr(args, "unity_base_port", 5005)),
            no_graphics=bool(getattr(args, "unity_no_graphics", True)),
            time_scale=float(getattr(args, "unity_time_scale", 20.0)),
            quality_level=int(getattr(args, "unity_quality_level", 1)),
            max_episode_steps=int(getattr(args, "time_limit", 1000)),
        )
    else:
        raise ValueError(
            f"Unknown env_backend='{backend}'. Use one of: dmc, gym, unity_mlagents."
        )

    env = env_wrapper.ActionRepeat(env, int(args.action_repeat))
    env = env_wrapper.NormalizeActions(env)
    repeat = max(1, int(args.action_repeat))
    duration = max(1, int(args.time_limit) // repeat)
    env = env_wrapper.TimeLimit(env, duration)
    return env


def preprocess_obs(obs: torch.Tensor) -> torch.Tensor:
    """Convert raw uint8 image observations to Dreamer float input space.

    Images are scaled from `[0, 255]` to roughly `[-0.5, 0.5]`, matching the
    RSSM encoder input expectations.
    """
    obs = obs.float()
    obs = obs / 255.0 - 0.5
    return obs


class Dreamer(nn.Module):
    """Dreamer world model for learning latent dynamics from pixels.

    Combines an RSSM for state representation, encoders/decoders for perception,
    and actor/critic networks for policy learning.
    """

    def __init__(
        self,
        args: Any,
        obs_shape: Tuple[int, ...],
        action_size: int,
        device: torch.device,
        restore: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.device = device

        # RSSM
        self.rssm = RSSM(
            action_size,
            args.stoch_size,
            args.deter_size,
            args.hidden_size,
            args.embedding_size,
            args.state_size,
            args.min_std,
            args.device,
        ).to(device)

        # Encoder/Decoder
        self.encoder = ConvEncoder(obs_shape, args.embedding_size).to(device)
        self.decoder = ConvDecoder(
            args.stoch_size + args.deter_size, args.embedding_size, obs_shape
        ).to(device)
        self.reward = DenseDecoder(
            args.stoch_size + args.deter_size, args.hidden_size, 1
        ).to(device)
        self.discount = DenseDecoder(
            args.stoch_size + args.deter_size, args.hidden_size, 1
        ).to(device)

        # Actor/Critic
        self.actor = ActionDecoder(
            args.stoch_size + args.deter_size, args.hidden_size, action_size
        ).to(device)
        self.value = DenseDecoder(
            args.stoch_size + args.deter_size, args.hidden_size, 1
        ).to(device)
        self.target_value = DenseDecoder(
            args.stoch_size + args.deter_size, args.hidden_size, 1
        ).to(device)
        self.target_value.load_state_dict(self.value.state_dict())

        # Optimizers
        self.world_model_opt = optim.Adam(
            [
                *self.encoder.parameters(),
                *self.decoder.parameters(),
                *self.reward.parameters(),
                *self.discount.parameters(),
                *self.rssm.parameters(),
            ],
            lr=args.world_lr,
            eps=1e-4,
            weight_decay=1e-6,
        )
        self.actor_opt = optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, eps=1e-4, weight_decay=1e-6
        )
        self.value_opt = optim.Adam(
            self.value.parameters(), lr=args.value_lr, eps=1e-4, weight_decay=1e-6
        )

        # Data buffer
        self.data_buffer = ReplayBuffer(
            args.buffer_size,
            obs_shape,
            action_size,
            device,
        )

        if restore:
            self.load(restore)

    def act_and_collect_data(self, env: Any, n_steps: int) -> np.ndarray:
        """Run agent in environment and collect experience."""
        episode_rewards = []
        obs = env.reset()
        obs = preprocess_obs(torch.tensor(obs, device=self.device)).unsqueeze(0)
        episode_reward = 0

        with torch.no_grad():
            for _ in range(n_steps):
                action = self.plan(obs, training=True)
                next_obs, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
                next_obs = preprocess_obs(
                    torch.tensor(next_obs, device=self.device)
                ).unsqueeze(0)
                episode_reward += reward

                self.data_buffer.add(obs, action, reward, done, next_obs)
                obs = next_obs

                if done:
                    episode_rewards.append(episode_reward)
                    obs = env.reset()
                    obs = preprocess_obs(
                        torch.tensor(obs, device=self.device)
                    ).unsqueeze(0)
                    episode_reward = 0

        if episode_reward > 0:
            episode_rewards.append(episode_reward)

        return np.array(episode_rewards)

    def collect_random_episodes(self, env: Any, n_episodes: int) -> np.ndarray:
        """Collect episodes with random actions for initial buffer population."""
        episode_rewards = []

        for _ in range(n_episodes):
            obs = env.reset()
            obs = preprocess_obs(torch.tensor(obs, device=self.device)).unsqueeze(0)
            episode_reward = 0
            done = False

            while not done:
                action = torch.randn(1, self.action_size, device=self.device)
                next_obs, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
                next_obs = preprocess_obs(
                    torch.tensor(next_obs, device=self.device)
                ).unsqueeze(0)
                episode_reward += reward

                self.data_buffer.add(obs, action, reward, done, next_obs)
                obs = next_obs

            episode_rewards.append(episode_reward)

        return np.array(episode_rewards)

    def plan(self, obs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Plan action using actor network."""
        with torch.no_grad():
            embed = self.encoder(obs)
            state = self.rssm.get_init_state(embed)
            action = self.actor(state).sample() if training else self.actor(state).mean
        return action

    def train_one_batch(self) -> Tuple[float, float, float]:
        """Train world model, actor, and critic for one batch."""
        obs, actions, rewards, discounts = self.data_buffer.sample(self.args.batch_size)

        # World model training
        with FreezeParameters([self.actor, self.value, self.target_value]):
            model_loss = self.train_world_model(obs, actions, rewards, discounts)

        # Actor/critic training
        with FreezeParameters(
            [self.encoder, self.decoder, self.reward, self.discount, self.rssm]
        ):
            actor_loss, value_loss = self.train_actor_critic(
                obs, actions, rewards, discounts
            )

        return model_loss, actor_loss, value_loss

    def train_world_model(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
    ) -> float:
        """Train RSSM and observation/reward models."""
        embed = self.encoder(obs)
        states, priors, posteriors = self.rssm.observe(embed, actions)

        obs_loss = self.decoder(states, obs).mean()
        reward_loss = self.reward(states, rewards).mean()
        discount_loss = self.discount(states, discounts).mean()
        kl_loss = self.rssm.kl_loss(priors, posteriors).mean()

        loss = obs_loss + reward_loss + discount_loss + self.args.kl_scale * kl_loss

        self.world_model_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_clip_norm)
        self.world_model_opt.step()

        return loss.item()

    def train_actor_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
    ) -> Tuple[float, float]:
        """Train actor and critic networks."""
        embed = self.encoder(obs)
        states, _, _ = self.rssm.observe(embed, actions)

        # Compute lambda returns
        with torch.no_grad():
            returns = compute_return(
                self.reward,
                self.discount,
                self.target_value,
                states,
                rewards,
                discounts,
                self.args,
            )

        # Actor loss
        actor_loss = -self.actor(states).log_prob(actions).mean()

        # Value loss
        value_loss = (self.value(states) - returns).pow(2).mean()

        # Update actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.args.grad_clip_norm
        )
        self.actor_opt.step()

        # Update critic
        self.value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value.parameters(), self.args.grad_clip_norm
        )
        self.value_opt.step()

        # Update target critic
        self.soft_update_target()

        return actor_loss.item(), value_loss.item()

    def soft_update_target(self) -> None:
        """Soft update target value network."""
        for target_param, param in zip(
            self.target_value.parameters(), self.value.parameters()
        ):
            target_param.data.copy_(
                self.args.target_update_tau * param.data
                + (1 - self.args.target_update_tau) * target_param.data
            )

    def evaluate(
        self, env: Any, n_episodes: int, render: bool = False
    ) -> Tuple[np.ndarray, list, Optional[torch.Tensor]]:
        """Evaluate agent performance."""
        episode_rewards = []
        video_frames = []

        for _ in range(n_episodes):
            obs = env.reset()
            obs = preprocess_obs(torch.tensor(obs, device=self.device)).unsqueeze(0)
            episode_reward = 0
            done = False
            episode_frames = []

            while not done:
                if render:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)

                action = self.plan(obs, training=False)
                next_obs, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
                next_obs = preprocess_obs(
                    torch.tensor(next_obs, device=self.device)
                ).unsqueeze(0)
                episode_reward += reward
                obs = next_obs

            episode_rewards.append(episode_reward)
            video_frames.append(episode_frames)

        return np.array(episode_rewards), video_frames, None

    def save(self, path: str) -> None:
        """Save model checkpoints."""
        torch.save(
            {
                "rssm": self.rssm.state_dict(),
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "reward": self.reward.state_dict(),
                "discount": self.discount.state_dict(),
                "actor": self.actor.state_dict(),
                "value": self.value.state_dict(),
                "target_value": self.target_value.state_dict(),
                "world_model_opt": self.world_model_opt.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "value_opt": self.value_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.rssm.load_state_dict(checkpoint["rssm"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.reward.load_state_dict(checkpoint["reward"])
        self.discount.load_state_dict(checkpoint["discount"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.value.load_state_dict(checkpoint["value"])
        self.target_value.load_state_dict(checkpoint["target_value"])
        self.world_model_opt.load_state_dict(checkpoint["world_model_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.value_opt.load_state_dict(checkpoint["value_opt"])
