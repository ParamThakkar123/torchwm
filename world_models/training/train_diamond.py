import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
import os

from world_models.configs.diamond_config import (
    DiamondConfig,
    HUMAN_SCORES,
    RANDOM_SCORES,
)
from world_models.envs.diamond_atari import make_diamond_atari_env
from world_models.datasets.diamond_dataset import ReplayBuffer, SequenceDataset
from world_models.models.diffusion.diamond_diffusion import (
    DiffusionUNet,
    EDMPreconditioner,
    EulerSampler,
)
from world_models.models.diffusion.reward_termination import (
    RewardTerminationModel,
    RewardTerminationLoss,
)
from world_models.models.diffusion.actor_critic import (
    ActorCriticNetwork,
    RLLoss,
)


class DiamondAgent:
    """
    DIAMOND: DIffusion As a Model Of eNvironment Dreams

    RL agent trained entirely within a diffusion world model.
    """

    def __init__(self, config: DiamondConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        self.env = make_diamond_atari_env(
            game=config.game,
            frameskip=config.frameskip,
            max_noop=config.max_noop,
            terminate_on_life_loss=config.terminate_on_life_loss,
            reward_clip=True,
            resize=(config.obs_size, config.obs_size),
            seed=config.seed,
        )

        self.action_dim = self.env.action_space.n

        self._build_models()

        self.replay_buffer = ReplayBuffer(
            capacity=100000,
            obs_shape=(config.obs_size, config.obs_size, 3),
            action_dim=1,
            device=config.device,
        )

        self.obs_history: List[np.ndarray] = []
        self.action_history: List[int] = []

        self.total_steps = 0
        self.global_step = 0

    def _build_models(self):
        """Initialize all DIAMOND models."""
        self.diffusion_model = DiffusionUNet(
            obs_channels=3,
            num_conditioning_frames=self.config.num_conditioning_frames,
            base_channels=self.config.diffusion_channels[0],
            channel_multipliers=tuple(self.config.diffusion_channels),
            num_res_blocks=self.config.diffusion_res_blocks,
            cond_dim=self.config.diffusion_cond_dim,
            action_dim=self.action_dim,
        ).to(self.device)

        self.edm_precond = EDMPreconditioner(
            sigma_data=self.config.sigma_data,
            p_mean=self.config.p_mean,
            p_std=self.config.p_std,
        )

        self.sampler = EulerSampler(
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            rho=self.config.rho,
            num_steps=self.config.num_sampling_steps,
        )

        self.reward_model = RewardTerminationModel(
            obs_channels=3,
            action_dim=self.action_dim,
            channels=tuple(self.config.reward_channels),
            lstm_dim=self.config.reward_lstm_dim,
            cond_dim=self.config.reward_cond_dim,
        ).to(self.device)

        self.reward_loss_fn = RewardTerminationLoss()

        self.actor_critic = ActorCriticNetwork(
            obs_channels=3,
            action_dim=self.action_dim,
            channels=tuple(self.config.actor_channels),
            lstm_dim=self.config.actor_lstm_dim,
        ).to(self.device)

        self.rl_loss_fn = RLLoss(
            discount_factor=self.config.discount_factor,
            lambda_returns=self.config.lambda_returns,
            entropy_weight=self.config.entropy_weight,
        )

        self.diffusion_opt = optim.AdamW(
            self.diffusion_model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay_diffusion,
        )

        self.reward_opt = optim.AdamW(
            self.reward_model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay_reward,
        )

        self.actor_opt = optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay_actor,
        )

    def _update_diffusion_model(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update diffusion world model."""
        self.diffusion_model.train()

        obs_seq = batch["obs_seq"]
        action_seq = batch["action_seq"]
        next_obs = batch["next_obs"]

        B, T, C, H, W = obs_seq.shape

        obs_history = obs_seq[:, :-1]
        target_obs = next_obs

        sigma = self.edm_precond.sample_noise_level(B, self.device)
        sigma = sigma.view(B, 1, 1, 1)

        noise = torch.randn_like(target_obs)
        noisy_target = target_obs + sigma * noise

        precond = self.edm_precond.get_preconditioners(sigma)
        model_input = precond["c_in"] * noisy_target

        model_output = self.diffusion_model(
            x=model_input,
            t=sigma.squeeze(-1).squeeze(-1),
            obs_history=obs_history,
            actions=action_seq[:, :-1],
        )

        target = (next_obs - precond["c_skip"] * noisy_target) / precond["c_out"]

        loss = F.mse_loss(model_output, target)

        self.diffusion_opt.zero_grad()
        loss.backward()
        self.diffusion_opt.step()

        return loss.item()

    def _update_reward_model(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update reward/termination model."""
        self.reward_model.train()

        obs_seq = batch["obs_seq"]
        action_seq = batch["action_seq"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        B, T, C, H, W = obs_seq.shape

        reward_logits, term_logits, _ = self.reward_model(
            obs=obs_seq,
            actions=action_seq,
        )

        total_loss, reward_loss, term_loss = self.reward_loss_fn(
            reward_logits=reward_logits[:, :-1],
            termination_logits=term_logits[:, :-1],
            rewards=rewards,
            terminated=dones,
        )

        self.reward_opt.zero_grad()
        total_loss.backward()
        self.reward_opt.step()

        return total_loss.item()

    def _update_actor_critic(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, float]:
        """Update actor-critic in imagination."""
        self.actor_critic.train()

        obs_seq = batch["obs_seq"]
        action_seq = batch["actions"]

        B, T, C, H, W = obs_seq.shape

        policy_logits, values, _ = self.actor_critic(obs_seq)

        lambda_returns = self.rl_loss_fn.compute_lambda_returns(
            rewards=action_seq.float(),
            values=values,
            dones=torch.zeros(B, T, dtype=torch.bool, device=self.device),
        )

        policy_loss = self.rl_loss_fn.policy_loss(
            policy_logits=policy_logits[:, :-1],
            actions=action_seq[:, :-1],
            lambda_returns=lambda_returns,
            values=values,
        )

        value_loss = self.rl_loss_fn.value_loss(
            values=values,
            lambda_returns=lambda_returns,
        )

        total_loss = policy_loss + value_loss

        self.actor_opt.zero_grad()
        total_loss.backward()
        self.actor_opt.step()

        return policy_loss.item(), value_loss.item()

    def _collect_experience(self, num_steps: int) -> List[float]:
        """Collect experience from the real environment."""
        rewards = []

        if len(self.obs_history) == 0:
            obs, _ = self.env.reset()
            obs = obs.astype(np.float32) / 255.0
            self.obs_history = [obs] * self.config.num_conditioning_frames

        for _ in range(num_steps):
            obs_tensor = (
                torch.from_numpy(
                    np.stack(self.obs_history[-self.config.num_conditioning_frames :])
                )
                .unsqueeze(0)
                .to(self.device)
            )

            action, _ = self.actor_critic.get_action(
                obs_tensor[0, -1],
                None,
                deterministic=False,
            )

            if random.random() < self.config.epsilon_greedy:
                action = self.env.action_space.sample()

            next_obs, reward, done, _ = self.env.step(action)

            next_obs = next_obs.astype(np.float32) / 255.0

            self.replay_buffer.add(
                obs=self.obs_history[-1].astype(np.uint8),
                action=action,
                reward=reward,
                done=done,
                next_obs=next_obs.astype(np.uint8),
            )

            rewards.append(reward)

            self.obs_history.append(next_obs)
            self.action_history.append(action)

            if done:
                obs, _ = self.env.reset()
                obs = obs.astype(np.float32) / 255.0
                self.obs_history = [obs] * self.config.num_conditioning_frames
                self.action_history = []

        return rewards

    @torch.no_grad()
    def _imagine_trajectory(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:
        """
        Imagine a trajectory using the diffusion world model.

        Args:
            obs_history: Initial observations [B, L, C, H, W]
            action_history: Initial actions [B, L]
            hidden_state: Initial LSTM hidden state

        Returns:
            obs_trajectory: [B, H, C, H, W]
            rewards: [B, H]
            dones: [B, H]
            hidden_state: Updated hidden state
        """
        B = obs_history.shape[0]
        horizon = self.config.imagination_horizon

        obs_trajectory = []
        rewards_list = []
        dones_list = []

        obs_current = obs_history
        actions_current = action_history

        for t in range(horizon):
            next_obs = self.sampler.sample(
                model=self.diffusion_model,
                shape=(B, 3, self.config.obs_size, self.config.obs_size),
                device=self.device,
                obs_history=obs_current,
                actions=actions_current,
            )

            reward, done, hidden_state = self.reward_model.predict(
                obs=next_obs[:, -1],
                actions=actions_current[:, -1],
                hidden_state=hidden_state,
            )

            obs_trajectory.append(next_obs)
            rewards_list.append(reward)
            dones_list.append(done)

            obs_current = torch.cat([obs_current[:, 1:], next_obs], dim=1)
            actions_current = torch.cat(
                [actions_current[:, 1:], reward.long().unsqueeze(-1)], dim=1
            )

        return (
            torch.stack(obs_trajectory, dim=1),
            torch.stack(rewards_list, dim=1),
            torch.stack(dones_list, dim=1),
            hidden_state,
        )

    def train(self):
        """Main training loop following Algorithm 1."""
        print(f"Training DIAMOND on {self.config.game}")
        print(f"Device: {self.device}")
        print(f"Action space: {self.action_dim}")

        for epoch in tqdm(range(self.config.num_epochs), desc="Training"):
            collected_rewards = self._collect_experience(
                self.config.environment_steps_per_epoch
            )

            if not self.replay_buffer.is_ready(self.config.batch_size):
                continue

            dataset = SequenceDataset(
                replay_buffer=self.replay_buffer,
                sequence_length=self.config.burn_in_length
                + self.config.imagination_horizon,
                burn_in=self.config.burn_in_length,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
            )

            diffusion_losses = []
            reward_losses = []
            policy_losses = []
            value_losses = []

            for _ in range(self.config.training_steps_per_epoch):
                batch = next(iter(dataloader))

                diffusion_loss = self._update_diffusion_model(batch)
                diffusion_losses.append(diffusion_loss)

                reward_loss = self._update_reward_model(batch)
                reward_losses.append(reward_loss)

                policy_loss, value_loss = self._update_actor_critic(batch)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

            if epoch % self.config.log_interval == 0:
                print(f"\nEpoch {epoch}:")
                print(f"  Diffusion loss: {np.mean(diffusion_losses):.4f}")
                print(f"  Reward loss: {np.mean(reward_losses):.4f}")
                print(f"  Policy loss: {np.mean(policy_losses):.4f}")
                print(f"  Value loss: {np.mean(value_losses):.4f}")
                print(f"  Collected reward: {np.mean(collected_rewards):.2f}")

            if epoch % self.config.eval_interval == 0:
                eval_reward = self.evaluate()
                hns = self._compute_human_normalized_score(eval_reward)
                print(f"  Eval reward: {eval_reward:.2f}, HNS: {hns:.3f}")

            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{epoch}.pt")

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 1) -> float:
        """Evaluate the agent."""
        self.actor_critic.eval()
        self.diffusion_model.eval()
        self.reward_model.eval()

        total_reward = 0.0

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            obs = obs.astype(np.float32) / 255.0

            obs_history = [obs] * self.config.num_conditioning_frames
            hidden_state = self.reward_model.init_hidden(1, self.device)

            done = False
            episode_reward = 0.0

            while not done:
                obs_tensor = (
                    torch.from_numpy(
                        np.stack(obs_history[-self.config.num_conditioning_frames :])
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )

                action, hidden_state = self.actor_critic.get_action(
                    obs_tensor[0, -1],
                    hidden_state,
                    deterministic=True,
                )

                next_obs, reward, done, _ = self.env.step(action)
                next_obs = next_obs.astype(np.float32) / 255.0

                episode_reward += reward

                obs_history.append(next_obs)

            total_reward += episode_reward

        return total_reward / num_episodes

    def _compute_human_normalized_score(self, score: float) -> float:
        """Compute human-normalized score."""
        game = self.config.game
        human = HUMAN_SCORES.get(game, 1.0)
        random = RANDOM_SCORES.get(game, 0.0)

        if human == random:
            return 0.0

        return (score - random) / (human - random)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs("checkpoints/diamond", exist_ok=True)
        checkpoint = {
            "config": self.config.__dict__,
            "diffusion_model": self.diffusion_model.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
            "diffusion_opt": self.diffusion_opt.state_dict(),
            "reward_opt": self.reward_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/diamond/{path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.diffusion_opt.load_state_dict(checkpoint["diffusion_opt"])
        self.reward_opt.load_state_dict(checkpoint["reward_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])


def train_diamond(game: str, seed: int = 0):
    """Train DIAMOND on a specific game."""
    config = DiamondConfig(
        game=game,
        seed=seed,
    )

    agent = DiamondAgent(config)
    agent.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="Breakout-v5")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_diamond(args.game, args.seed)
