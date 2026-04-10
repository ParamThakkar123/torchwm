"""
PyBullet to DIAMOND World Model Training Script

This script trains the DIAMOND diffusion world model on PyBullet-generated environments.
It wraps the PyBullet BasicEnv with GymWrapperEnv and adapts actions for DIAMOND.

==============================================================================
USAGE
==============================================================================

Train with default settings (2 epochs):
    python examples/pybullet_diamond_train.py

Train for more epochs:
    python examples/pybullet_diamond_train.py --num_epochs 10

Custom number of discrete actions:
    python examples/pybullet_diamond_train.py --num_actions 21

Use specific device:
    python examples/pybullet_diamond_train.py --device cuda

==============================================================================
ARGUMENTS
==============================================================================

    --num_actions : Number of discrete actions for the agent (default: 11)
    --seed       : Random seed (default: 0)
    --preset     : Model preset - small, medium, or large (default: small)
    --device     : Device to run on - cuda or cpu (default: auto-detect)
    --num_epochs : Number of training epochs (default: 100)

==============================================================================
WHAT IT DOES
==============================================================================

1. Creates a PyBullet environment with a simple box that can be pushed
2. Wraps it in a gym-compatible interface with discrete action space
3. Trains three DIAMOND components:
   - Diffusion model: Predicts next observation given history + action
   - Reward model: Predicts reward and done flags
   - Actor-critic: Learns policy using imagined trajectories

4. Uses imagination-based RL:
   - Collects some real experience
   - Uses world model to generate imagined trajectories
   - Updates policy using imagined data

==============================================================================
OUTPUT
==============================================================================

Checkpoints saved to: checkpoints/diamond_pybullet/
  - checkpoint_0.pt, checkpoint_1.pt, etc.

Logs show:
  - Diffusion loss (world model prediction error)
  - Reward loss (reward prediction error)
  - Policy loss (policy gradient)
  - Value loss (value function error)
  - Collected reward (real environment reward)

==============================================================================
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces

from world_models.configs.diamond_config import DiamondConfig
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


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function that converts (T, H, W, C) -> (T, C, H, W)."""
    obs_seq = torch.stack([item["obs_seq"] for item in batch])
    action_seq = torch.stack([item["action_seq"] for item in batch])
    next_obs = torch.stack([item["next_obs"] for item in batch])
    rewards = torch.stack([item["rewards"] for item in batch])
    dones = torch.stack([item["dones"] for item in batch])
    actions = torch.stack([item["actions"] for item in batch])

    # Convert from (B, T, H, W, C) to (B, T, C, H, W)
    obs_seq = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
    # Convert from (B, H, W, C) to (B, C, H, W)
    next_obs = next_obs.permute(0, 3, 1, 2).contiguous()

    return {
        "obs_seq": obs_seq,
        "action_seq": action_seq,
        "next_obs": next_obs,
        "rewards": rewards,
        "dones": dones,
        "actions": actions,  # Added for actor-critic
    }


class PyBulletDiamondWrapper(gym.Wrapper):
    """
    Wrapper that adapts PyBullet environment for DIAMOND.
    Discretizes continuous actions and provides proper reward/shape.
    """

    def __init__(self, pybullet_env, num_actions: int = 11):
        super().__init__(pybullet_env)
        self.num_actions = num_actions

        # Map discrete actions to continuous torque range [-1, 1]
        self.action_map = np.linspace(-1.0, 1.0, num_actions)

        # Convert continuous Box to discrete Discrete
        self.action_space = spaces.Discrete(num_actions)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Convert discrete action to continuous torque and step."""
        continuous_action = self.action_map[action]

        # Get the underlying continuous action space bounds
        original_action = np.array([continuous_action], dtype=np.float32)

        obs, reward, done, info = self.env.step(original_action)

        # Design a simple reward based on task
        custom_reward = self._compute_reward(obs, done, reward)

        return obs, custom_reward, done, info

    def _compute_reward(self, obs: np.ndarray, done: bool, env_reward: float) -> float:
        """Compute custom reward based on observation."""
        if done:
            return -1.0

        # Use the environment reward if available, otherwise compute based on position
        # For a falling box, we can use position-based reward
        # Higher = better (box should try to stay up)
        try:
            # Get position from physics - approximate from pixel data
            # Since we don't have direct access, use a simple heuristic
            # If the box is visible in frame (not fallen too far), give positive reward
            return env_reward if env_reward != 0 else 0.1
        except:
            return 0.0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        return obs, info


def create_pybullet_env(config: Dict, num_actions: int = 11) -> PyBulletDiamondWrapper:
    """Create a PyBullet environment wrapped for DIAMOND."""
    from torchwm.sim.envs.basic_env import BasicEnv
    from torchwm.sim.gym_wrapper import GymWrapperEnv

    pybullet_config = {
        "physics": {
            "timestep": 1 / 60,
            "substeps": 1,
            "num_solver_iterations": 50,
            "gravity_z": -9.81,
        },
        "generator": {
            "objects": [
                {
                    "shape": {"type": "box", "size": [0.5, 0.5, 0.5]},
                    "position": [0, 0, 1],
                    "mass": 1.0,
                }
            ]
        },
        "camera": {
            "width": 64,
            "height": 64,
            "fov": 60,
            "position": [1, 1, 1],
            "target": [0, 0, 0],
        },
    }

    base_env = BasicEnv(pybullet_config)
    gym_env = GymWrapperEnv(
        base_env,
        sensors=["camera"],
        action_config={"type": "torque", "body_index": 0, "low": -1.0, "high": 1.0},
    )

    return PyBulletDiamondWrapper(gym_env, num_actions=num_actions)


class PyBulletDiamondAgent:
    """
    DIAMOND agent adapted for PyBullet environments.
    """

    def __init__(self, config: DiamondConfig, num_actions: int = 11):
        self.config = config
        self.num_actions = num_actions
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        self.env = create_pybullet_env({}, num_actions)
        self.action_dim = self.num_actions

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

        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0

    def _build_models(self):
        """Initialize all DIAMOND models."""
        # Use base_channels=32 which with [1,1,1,1] multipliers gives 32 channels everywhere
        base_ch = 32
        channel_mults = tuple(self.config.diffusion_channels)

        self.diffusion_model = DiffusionUNet(
            obs_channels=3,
            num_conditioning_frames=self.config.num_conditioning_frames,
            base_channels=base_ch,
            channel_multipliers=channel_mults,
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

        obs_seq = batch["obs_seq"]  # (B, T, C, H, W)
        action_seq = batch["action_seq"]  # (B, T)
        next_obs = batch["next_obs"]  # (B, C, H, W)

        B, T, C, H, W = obs_seq.shape
        num_cond = self.config.num_conditioning_frames

        # Use LAST num_cond frames as conditioning
        obs_history = obs_seq[:, T - num_cond :]  # (B, num_cond, C, H, W)

        # Flatten obs_history to (B, num_cond*C, H, W) - the model expects 4D
        obs_history_flat = obs_history.view(B, num_cond * C, H, W).contiguous()

        actions_for_cond = action_seq[:, T - num_cond :]  # (B, num_cond)
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
            obs_history=obs_history_flat,
            actions=actions_for_cond,
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

        obs_seq = batch["obs_seq"]  # (B, T, C, H, W)
        action_seq = batch["action_seq"]  # (B, T)
        rewards = batch["rewards"]  # (B, T)
        dones = batch["dones"]  # (B, T)

        B, T, C, H, W = obs_seq.shape

        # The reward model expects (B, T, C, H, W) and (B, T)
        reward_logits, term_logits, _ = self.reward_model(
            obs=obs_seq,
            actions=action_seq,
        )

        # Convert rewards to classification targets (-1, 0, 1 -> 0, 1, 2)
        # Clip rewards to [-1, 1] range
        reward_targets = torch.clamp(rewards, -1, 1).long() + 1  # -> 0, 1, 2

        # Convert dones to binary (0 or 1)
        term_targets = dones.long()

        # Compute loss - use cross entropy
        reward_loss = F.cross_entropy(
            reward_logits.view(-1, 3), reward_targets.view(-1)
        )

        term_loss = F.cross_entropy(term_logits.view(-1, 2), term_targets.view(-1))

        total_loss = reward_loss + term_loss

        self.reward_opt.zero_grad()
        total_loss.backward()
        self.reward_opt.step()

        return total_loss.item()

    def _update_actor_critic_imagination(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, float]:
        """Update actor-critic using imagined trajectories from the world model."""
        self.actor_critic.train()

        obs_seq = batch["obs_seq"]  # (B, T, C, H, W)
        action_seq = batch["actions"]  # (B, T, 1)
        action_seq = action_seq.squeeze(-1)  # (B, T)

        B, T, C, H, W = obs_seq.shape
        num_cond = self.config.num_conditioning_frames

        obs_history = obs_seq[:, :num_cond]  # (B, num_cond, C, H, W)
        actions_init = action_seq[:, :num_cond]  # (B, num_cond)

        hidden_state = self.reward_model.init_hidden(B, self.device)

        obs_traj, rewards_imag, dones_imag, hidden_state = self._imagine_trajectory(
            obs_history, actions_init, hidden_state
        )

        obs_full = torch.cat([obs_history, obs_traj], dim=1)  # (B, num_cond+H, C, H, W)

        policy_logits, values, _ = self.actor_critic(obs_full)

        H = self.config.imagination_horizon
        T_imag = num_cond + H

        rewards_imag = rewards_imag.squeeze(-1)  # (B, H)
        dones_imag = dones_imag.squeeze(-1).squeeze(-1)  # (B, H)

        rewards_full = torch.cat(
            [torch.zeros(B, num_cond, device=self.device), rewards_imag], dim=1
        )
        dones_full = torch.cat(
            [
                torch.zeros(B, num_cond, dtype=torch.bool, device=self.device),
                dones_imag,
            ],
            dim=1,
        )

        gamma = self.config.discount_factor
        lam = self.config.lambda_returns

        lambda_returns = torch.zeros(B, T_imag, device=self.device)
        bootstrap = values[:, -1, 0]
        returns = bootstrap

        for t in reversed(range(T_imag)):
            mask = 1.0 - dones_full[:, t].float()
            returns = rewards_full[:, t] + gamma * mask * returns
            lambda_returns[:, t] = returns

        log_probs = F.log_softmax(policy_logits, dim=-1)
        actions_imag = action_seq[:, :T_imag].long()

        log_probs_flat = log_probs.clone().reshape(B * T_imag, self.action_dim)
        actions_flat = actions_imag.clone().reshape(B * T_imag)

        selected_log_probs = (
            torch.gather(log_probs_flat, dim=1, index=actions_flat.unsqueeze(-1))
            .squeeze(-1)
            .view(B, T_imag)
        )

        # Use all values including bootstrap for advantages
        values_for_adv = values[:, :, 0]  # (B, T_imag)
        advantages = lambda_returns - values_for_adv.detach()

        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values_for_adv, lambda_returns.detach())

        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.config.entropy_weight * entropy

        total_loss = policy_loss + 0.5 * value_loss + entropy_loss

        self.actor_opt.zero_grad()
        total_loss.backward()
        self.actor_opt.step()

        return policy_loss.item(), value_loss.item()

    def _update_actor_critic(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, float]:
        """Update actor-critic using the batch data (real experience)."""
        self.actor_critic.train()

        obs_seq = batch["obs_seq"]  # (B, T, C, H, W)
        action_seq = batch["actions"]  # (B, T, 1)
        rewards = batch["rewards"]  # (B, T)
        dones = batch["dones"]  # (B, T)

        B, T, C, H, W = obs_seq.shape

        action_seq = action_seq.squeeze(-1)

        policy_logits, values, _ = self.actor_critic(obs_seq)

        with torch.no_grad():
            last_obs = obs_seq[:, -1:]
            _, bootstrap_value, _ = self.actor_critic(last_obs)

        values_with_bootstrap = torch.cat([values, bootstrap_value], dim=1)

        gamma = self.config.discount_factor
        lam = self.config.lambda_returns

        rewards_clipped = torch.clamp(rewards, -10, 10)
        lambda_returns = torch.zeros_like(rewards)

        cumulative_return = bootstrap_value.squeeze(-1).squeeze(-1)
        for t in reversed(range(T)):
            mask = 1.0 - dones[:, t].float()
            delta = rewards_clipped[:, t] + gamma * mask * cumulative_return
            cumulative_return = delta + gamma * lam * mask * cumulative_return
            lambda_returns[:, t] = cumulative_return

        log_probs = F.log_softmax(policy_logits, dim=-1)
        actions = action_seq.long()

        log_probs_flat = log_probs.clone().reshape(B * T, self.action_dim)
        actions_flat = actions.clone().reshape(B * T)

        selected_log_probs = (
            torch.gather(log_probs_flat, dim=1, index=actions_flat.unsqueeze(-1))
            .squeeze(-1)
            .view(B, T)
        )

        values_for_adv = values_with_bootstrap[:, :-1, 0]
        advantages = lambda_returns - values_for_adv.detach()

        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values_for_adv, lambda_returns.detach())

        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.config.entropy_weight * entropy

        total_loss = policy_loss + 0.5 * value_loss + entropy_loss

        self.actor_opt.zero_grad()
        total_loss.backward()
        self.actor_opt.step()

        return policy_loss.item(), value_loss.item()

    def _collect_experience(self, num_steps: int) -> List[float]:
        """Collect experience from the real environment."""
        rewards = []

        if len(self.obs_history) == 0:
            obs, _ = self.env.reset()
            # Keep as (H, W, C) for replay buffer
            obs = obs.astype(np.float32) / 255.0
            self.obs_history = [obs] * self.config.num_conditioning_frames

        for _ in range(num_steps):
            # obs_tensor shape: (1, T, H, W, C)
            obs_tensor = (
                torch.from_numpy(
                    np.stack(self.obs_history[-self.config.num_conditioning_frames :])
                )
                .unsqueeze(0)
                .to(self.device)
            )

            # Convert to (B, T, C, H, W) for the network
            obs_for_network = obs_tensor.permute(0, 1, 4, 2, 3)

            # Pass last frame with batch dim (B, C, H, W) = (1, C, H, W)
            action, _ = self.actor_critic.get_action(
                obs_for_network[0, -1].unsqueeze(0),
                None,
                deterministic=False,
            )

            if random.random() < self.config.epsilon_greedy:
                action = self.env.action_space.sample()

            next_obs, reward, done, _ = self.env.step(action)

            # Keep as (H, W, C) for replay buffer
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
                # Keep as (H, W, C)
                obs = obs.astype(np.float32) / 255.0
                self.obs_history = [obs] * self.config.num_conditioning_frames
                self.action_history = []

        # Update running reward statistics
        if len(rewards) > 0:
            rewards_np = np.array(rewards)
            batch_mean = rewards_np.mean()
            batch_var = rewards_np.var()
            batch_count = len(rewards_np)

            # Welford's online algorithm
            delta = batch_mean - self.reward_mean
            self.reward_mean += delta * batch_count / (self.reward_count + batch_count)
            self.reward_var = (
                self.reward_count * self.reward_var + batch_count * batch_var
            ) / (self.reward_count + batch_count)
            self.reward_count += batch_count

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
        """Imagine a trajectory using the diffusion world model."""
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

            next_obs_4d = next_obs  # (B, C, H, W)
            reward, done, hidden_state = self.reward_model.predict(
                obs=next_obs_4d,
                actions=actions_current[:, -1],
                hidden_state=hidden_state,
            )

            obs_trajectory.append(next_obs_4d)
            rewards_list.append(reward)
            dones_list.append(done)

            next_obs_seq = next_obs.unsqueeze(1)  # (B, 1, C, H, W)
            random_actions = torch.randint(
                0, self.action_dim, (B, 1), device=self.device
            )
            obs_current = torch.cat([obs_current[:, 1:], next_obs_seq], dim=1)
            actions_current = torch.cat([actions_current[:, 1:], random_actions], dim=1)

        return (
            torch.stack(obs_trajectory, dim=1),
            torch.stack(rewards_list, dim=1),
            torch.stack(dones_list, dim=1),
            hidden_state,
        )

    def train(self):
        """Main training loop."""
        print("[train] Starting...")
        print(f"Training DIAMOND on PyBullet Environment")
        print(f"Device: {self.device}")
        print(f"Action space: {self.action_dim}")

        for epoch in range(self.config.num_epochs):
            print(f"\n=== Epoch {epoch} ===")
            print("Collecting experience...")
            collected_rewards = self._collect_experience(
                self.config.environment_steps_per_epoch
            )
            print(
                f"Collected {len(collected_rewards)} rewards, buffer size: {self.replay_buffer.size}"
            )

            if not self.replay_buffer.is_ready(self.config.batch_size):
                print(
                    f"Buffer not ready (need {self.config.batch_size}), skipping training"
                )
                continue

            print("Creating dataset...")
            dataset = SequenceDataset(
                replay_buffer=self.replay_buffer,
                sequence_length=self.config.burn_in_length
                + self.config.imagination_horizon,
                burn_in=self.config.burn_in_length,
            )
            print(f"Dataset created with {len(dataset)} sequences")

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn,
            )

            diffusion_losses = []
            reward_losses = []
            policy_losses = []
            value_losses = []

            print(f"Running {self.config.training_steps_per_epoch} training steps...")
            for step in range(self.config.training_steps_per_epoch):
                batch = next(iter(dataloader))

                diffusion_loss = self._update_diffusion_model(batch)
                diffusion_losses.append(diffusion_loss)

                reward_loss = self._update_reward_model(batch)
                reward_losses.append(reward_loss)

                policy_loss, value_loss = self._update_actor_critic_imagination(batch)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

                if step % 10 == 0:
                    print(f"  Step {step}/{self.config.training_steps_per_epoch}")

            print(f"\nEpoch {epoch}:")
            print(f"  Diffusion loss: {np.mean(diffusion_losses):.4f}")
            print(f"  Reward loss: {np.mean(reward_losses):.4f}")
            print(f"  Policy loss: {np.mean(policy_losses):.4f}")
            print(f"  Value loss: {np.mean(value_losses):.4f}")
            print(f"  Collected reward: {np.mean(collected_rewards):.2f}")

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
            # Keep as (H, W, C)
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

                # Convert to (B, T, C, H, W)
                obs_for_network = obs_tensor.permute(0, 1, 4, 2, 3)

                action, hidden_state = self.actor_critic.get_action(
                    obs_for_network[0, -1].unsqueeze(0),
                    hidden_state,
                    deterministic=True,
                )

                next_obs, reward, done, _ = self.env.step(action)
                # Keep as (H, W, C)
                next_obs = next_obs.astype(np.float32) / 255.0

                episode_reward += reward

                obs_history.append(next_obs)

            total_reward += episode_reward

        return total_reward / num_episodes

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs("checkpoints/diamond_pybullet", exist_ok=True)
        checkpoint = {
            "config": self.config.__dict__,
            "diffusion_model": self.diffusion_model.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
            "diffusion_opt": self.diffusion_opt.state_dict(),
            "reward_opt": self.reward_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/diamond_pybullet/{path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.diffusion_opt.load_state_dict(checkpoint["diffusion_opt"])
        self.reward_opt.load_state_dict(checkpoint["reward_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])


def train_diamond_on_pybullet(
    num_actions: int = 11,
    seed: int = 0,
    preset: Optional[str] = "small",
    device: str = "cuda",
    num_epochs: int = 100,
):
    """Train DIAMOND on PyBullet environment."""
    print("Creating config...")
    # For "small" preset, use smaller multipliers to keep channels reasonable
    # Channel multipliers should be like (1, 2, 4, 8) not actual channel counts
    config = DiamondConfig(
        game="PyBullet-Box",
        seed=seed,
        preset=None,  # Don't use preset - set manually
        device=device,
        num_epochs=num_epochs,
        obs_size=64,
        batch_size=16,
        training_steps_per_epoch=100,
        environment_steps_per_epoch=50,
    )

    # Manually set model config with channel multipliers
    # base_channels=32, multipliers (1,1,1,1) gives 32, 32, 32, 32 channels
    # All divisible by 32 (GroupNorm group size)
    config.diffusion_channels = [1, 1, 1, 1]  # multipliers, NOT channel counts
    config.diffusion_res_blocks = 2
    config.diffusion_cond_dim = 128
    config.reward_channels = [8, 16, 32, 32]
    config.reward_lstm_dim = 256
    config.actor_channels = [8, 16, 32, 32]
    config.actor_lstm_dim = 256

    # Tuned RL hyperparameters for stable training
    config.discount_factor = 0.99
    config.lambda_returns = 0.95
    config.entropy_weight = 0.01

    agent = PyBulletDiamondAgent(config, num_actions=num_actions)
    agent.train()

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_actions", type=int, default=11, help="Number of discrete actions"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preset", type=str, default="small", choices=["small", "medium", "large"]
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_epochs", type=int, default=100)
    args = parser.parse_args()

    agent = train_diamond_on_pybullet(
        num_actions=args.num_actions,
        seed=args.seed,
        preset=args.preset,
        device=args.device,
        num_epochs=args.num_epochs,
    )

    print("Training complete!")
