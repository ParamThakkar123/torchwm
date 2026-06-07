import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import os
from pathlib import Path
import argparse

from world_models.configs.diamond_config import (
    DiamondConfig,
    HUMAN_SCORES,
    RANDOM_SCORES,
)
from world_models.envs.diamond_atari import make_diamond_atari_env
from gym.spaces import Discrete, Box
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

        # action_space may be different Space types; prefer Discrete.n when available
        # Declare attribute type for static checkers
        self.action_dim: int = 0

        if isinstance(self.env.action_space, Discrete):
            self.action_dim = int(self.env.action_space.n)
        elif isinstance(self.env.action_space, Box):
            # continuous action space -> flatten dimensions
            shape = getattr(self.env.action_space, "shape", None)
            if shape is None:
                raise TypeError("Box action_space has no shape")
            # ensure shape is numeric sequence before taking product
            self.action_dim = int(np.prod(tuple(shape)))
        else:
            # fallback: try to read 'n' attribute, else raise informative error
            if hasattr(self.env.action_space, "n"):
                self.action_dim = int(getattr(self.env.action_space, "n"))
            else:
                raise TypeError(
                    f"Unsupported action_space type: {type(self.env.action_space)}"
                )

        self._build_models()

        self.replay_buffer = ReplayBuffer(
            capacity=100000,
            obs_shape=(config.obs_size, config.obs_size, 3),
            action_dim=1,
            device=config.device,
        )

        self.obs_history: List[np.ndarray] = []
        # keep a raw-uint8 history in parallel with the normalized history
        self.obs_history_raw: List[np.ndarray] = []
        self.action_history: List[int] = []

        self.total_steps = 0
        self.global_step = 0
        # last LSTM hidden states (saved for reproducible imagined rollouts)
        self.last_policy_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.last_reward_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _build_models(self):
        """Initialize all DIAMOND models."""
        self.diffusion_model = DiffusionUNet(
            obs_channels=3,
            num_conditioning_frames=self.config.num_conditioning_frames,
            # config.diffusion_channels is a list of absolute channel sizes per level
            # DiffusionUNet expects base_channels and channel_multipliers (multipliers
            # relative to base). Convert absolute sizes to multipliers here.
            base_channels=self.config.diffusion_channels[0],
            channel_multipliers=tuple(
                [
                    int(c // self.config.diffusion_channels[0])
                    for c in self.config.diffusion_channels
                ]
            ),
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
            edm_precond=self.edm_precond,
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

        # use only the last `num_conditioning_frames` for conditioning
        obs_history = obs_seq[:, -self.config.num_conditioning_frames :]
        target_obs = next_obs

        sigma = self.edm_precond.sample_noise_level(B, self.device)
        sigma = sigma.view(B, 1, 1, 1)

        noise = torch.randn_like(target_obs)
        noisy_target = target_obs + sigma * noise

        precond = self.edm_precond.get_preconditioners(sigma)
        model_input = precond["c_in"] * noisy_target

        # use c_noise (log-sigma transform) for time conditioning as in EDM
        t_cond = precond["c_noise"].squeeze(-1).squeeze(-1)

        # Debug asserts: ensure shapes are as expected
        try:
            # obs_seq: [B, T, C, H, W], obs_history: [B, L, C, H, W], next_obs [B, C, H, W]
            assert obs_seq.ndim == 5
            assert obs_history.ndim == 5
            assert target_obs.ndim == 4
        except AssertionError:
            print(
                f"DEBUG SHAPES: obs_seq={getattr(obs_seq, 'shape', None)}, obs_history={getattr(obs_history, 'shape', None)}, target_obs={getattr(target_obs, 'shape', None)}"
            )

        model_output = self.diffusion_model(
            x=model_input,
            t=t_cond,
            obs_history=obs_history,
            actions=action_seq[:, -self.config.num_conditioning_frames :],
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

        # Align target sequence lengths: reward_logits has same temporal length as obs_seq
        # We predict next-step rewards for all but the last conditioning frame.
        # Align rewards/dones length with reward_logits[:, :-1]
        T_logits = reward_logits.shape[1]
        target_len = max(0, T_logits - 1)
        rewards_target = rewards[:, :target_len]
        dones_target = dones[:, :target_len]

        total_loss, reward_loss, term_loss = self.reward_loss_fn(
            reward_logits=reward_logits[:, :-1],
            termination_logits=term_logits[:, :-1],
            rewards=rewards_target,
            terminated=dones_target,
        )

        self.reward_opt.zero_grad()
        total_loss.backward()
        self.reward_opt.step()

        return total_loss.item()

    def _update_actor_critic(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, float]:
        """Update actor-critic using imagined rollouts.

        This replaces using real dataset trajectories for policy/value updates
        and follows the paper: compute policy and value losses on imagined
        trajectories produced by the diffusion world model.
        """
        self.actor_critic.train()

        obs_seq = batch["obs_seq"]
        action_seq = batch.get("action_seq", batch.get("actions"))

        B, seq_T, C, H, W = obs_seq.shape

        # Determine burn-in / conditioning length and horizon
        burn_in = self.config.burn_in_length
        horizon = self.config.imagination_horizon

        # safety checks
        assert seq_T >= burn_in, "Sequence shorter than burn-in"

        # initial conditioning sequences for imagination
        obs_history = obs_seq[:, :burn_in]
        if action_seq is None:
            action_history = torch.zeros(
                (B, burn_in), dtype=torch.long, device=self.device
            )
        else:
            action_history = action_seq[:, :burn_in]

        # init reward model hidden state for batched imagination. Prefer a
        # restored last_reward_hidden when available (broadcast if needed).
        reward_hidden = None
        if self.last_reward_hidden is not None:
            try:
                h_saved, c_saved = self.last_reward_hidden
                saved_B = h_saved.shape[1]
                if saved_B == B:
                    reward_hidden = (h_saved.to(self.device), c_saved.to(self.device))
                elif saved_B == 1:
                    reward_hidden = (
                        h_saved.to(self.device).repeat(1, B, 1),
                        c_saved.to(self.device).repeat(1, B, 1),
                    )
                else:
                    reward_hidden = None
            except Exception:
                reward_hidden = None

        if reward_hidden is None:
            reward_hidden = self.reward_model.init_hidden(B, self.device)

        # prime the policy LSTM with the burn-in observations to obtain a
        # proper initial hidden state for imagined rollouts. If a saved
        # last_policy_hidden exists (restored from checkpoint), prefer it so
        # imagined rollouts can be exactly reproduced across runs. If the
        # saved hidden state's batch dimension is 1, broadcast it to current
        # batch size.
        policy_hidden_init = None
        if self.last_policy_hidden is not None:
            try:
                h_saved, c_saved = self.last_policy_hidden
                saved_B = h_saved.shape[1]
                if saved_B == B:
                    policy_hidden_init = (
                        h_saved.to(self.device),
                        c_saved.to(self.device),
                    )
                elif saved_B == 1:
                    policy_hidden_init = (
                        h_saved.to(self.device).repeat(1, B, 1),
                        c_saved.to(self.device).repeat(1, B, 1),
                    )
                else:
                    policy_hidden_init = None
            except Exception:
                policy_hidden_init = None

        if policy_hidden_init is None:
            with torch.no_grad():
                _, _, policy_hidden_init = self.actor_critic(obs_history)
            # store a CPU copy so checkpoints are device independent
            try:
                self.last_policy_hidden = (
                    policy_hidden_init[0].detach().cpu(),
                    policy_hidden_init[1].detach().cpu(),
                )
            except Exception:
                self.last_policy_hidden = None

        # Imagine a trajectory and obtain policy actions taken during imagination
        (
            obs_imag,
            rewards_imag,
            dones_imag,
            policy_actions_imag,
            reward_hidden,
        ) = self._imagine_trajectory(
            obs_history, action_history, reward_hidden, policy_hidden=policy_hidden_init
        )

        # store the final reward hidden state (CPU) for checkpointing / replay
        try:
            self.last_reward_hidden = (
                reward_hidden[0].detach().cpu(),
                reward_hidden[1].detach().cpu(),
            )
        except Exception:
            self.last_reward_hidden = None

        # Compute policy logits and values for imagined observations using the
        # same initial policy hidden state used during imagination so the
        # logits align with the sampled actions.
        policy_logits, values, _ = self.actor_critic(obs_imag, policy_hidden_init)

        # values: [B, H, 1] -> squeeze to [B, H]
        values_squeezed = values.squeeze(-1)
        # append bootstrap last value to make [B, H+1]
        values_with_bootstrap = torch.cat(
            [values_squeezed, values_squeezed[:, -1:].detach()], dim=1
        )

        # lambda-returns on imagined rewards
        lambda_returns = self.rl_loss_fn.compute_lambda_returns(
            rewards=rewards_imag,
            values=values_with_bootstrap,
            dones=dones_imag,
        )

        policy_loss = self.rl_loss_fn.policy_loss(
            policy_logits=policy_logits,
            actions=policy_actions_imag,
            lambda_returns=lambda_returns,
            values=values_with_bootstrap,
        )

        value_loss = self.rl_loss_fn.value_loss(
            values=values_with_bootstrap.unsqueeze(-1),
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
            raw_obs, _ = self.env.reset()
            norm_obs = raw_obs.astype(np.float32) / 255.0
            # maintain both normalized and raw histories
            self.obs_history = [norm_obs] * self.config.num_conditioning_frames
            self.obs_history_raw = [raw_obs] * self.config.num_conditioning_frames

        for _ in range(num_steps):
            # build tensor [1, L, C, H, W] with channels-first
            obs_np = np.stack(self.obs_history[-self.config.num_conditioning_frames :])
            # obs_np: [L, H, W, C] -> transpose to [L, C, H, W]
            obs_np = obs_np.transpose(0, 3, 1, 2)
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

            # pass a batched single observation [1, C, H, W]
            action, _ = self.actor_critic.get_action(
                obs_tensor[:, -1],
                None,
                deterministic=False,
            )

            if random.random() < self.config.epsilon_greedy:
                action = self.env.action_space.sample()

            next_obs, reward, done, _ = self.env.step(action)

            # env typically returns uint8 frames; keep raw and normalized
            next_obs_raw = next_obs
            next_obs = next_obs_raw.astype(np.float32) / 255.0

            # store raw uint8 frames in the replay buffer (avoid lossy casts)
            self.replay_buffer.add(
                obs=self.obs_history_raw[-1],
                action=action,
                reward=reward,
                done=done,
                next_obs=next_obs_raw,
            )

            rewards.append(reward)

            # update both normalized and raw histories
            self.obs_history.append(next_obs)
            self.obs_history_raw.append(next_obs_raw)
            self.action_history.append(action)

            if done:
                raw_obs, _ = self.env.reset()
                norm_obs = raw_obs.astype(np.float32) / 255.0
                self.obs_history = [norm_obs] * self.config.num_conditioning_frames
                self.obs_history_raw = [raw_obs] * self.config.num_conditioning_frames
                self.action_history = []

        return rewards

    @torch.no_grad()
    def _imagine_trajectory(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        policy_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
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

        # initialize a policy hidden state for batched policy sampling during imagination
        # allow caller to provide an initial policy hidden state (primed by
        # burn-in sequence); otherwise initialize fresh hidden state
        if policy_hidden is None:
            policy_hidden = self.actor_critic.init_hidden(B, self.device)

        policy_actions_list = []

        for t in range(horizon):
            # sampler returns [B, C, H, W]
            sampled = self.sampler.sample(
                model=self.diffusion_model,
                shape=(B, 3, self.config.obs_size, self.config.obs_size),
                device=self.device,
                obs_history=obs_current,
                actions=actions_current,
            )

            # predict reward/termination from the sampled frame [B, C, H, W]
            reward, done, hidden_state = self.reward_model.predict(
                obs=sampled,
                actions=actions_current[:, -1],
                hidden_state=hidden_state,
            )

            # append squeezed frame [B, C, H, W] for stacking later
            obs_trajectory.append(sampled)
            rewards_list.append(reward)
            dones_list.append(done)

            # update conditioning sequences: obs_current expects [B, L, C, H, W]
            next_obs_seq = sampled.unsqueeze(1)
            obs_current = torch.cat([obs_current[:, 1:], next_obs_seq], dim=1)

            # Batch-query the policy for the next actions for all samples at once
            # Maintain and update the policy LSTM hidden state across imagined
            # timesteps so the policy can condition on the imagined trajectory.
            policy_actions, policy_hidden = self.actor_critic.get_actions(
                sampled, policy_hidden, deterministic=False
            )

            # collect the actions (B,) per timestep
            policy_actions_list.append(policy_actions)

            # policy_actions: [B] -> make [B, 1] so it can be concatenated
            policy_actions = policy_actions.unsqueeze(-1)
            actions_current = torch.cat([actions_current[:, 1:], policy_actions], dim=1)

        return (
            torch.stack(obs_trajectory, dim=1),
            torch.stack(rewards_list, dim=1),
            torch.stack(dones_list, dim=1),
            torch.stack(policy_actions_list, dim=1),
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

            # iterate over the dataloader properly; avoid recreating iterator each step
            data_iter = iter(dataloader)
            for _ in range(self.config.training_steps_per_epoch):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

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
            # initialize separate hidden states for reward model and policy
            reward_hidden = self.reward_model.init_hidden(1, self.device)
            policy_hidden = self.actor_critic.init_hidden(1, self.device)

            done = False
            episode_reward = 0.0

            while not done:
                obs_np = np.stack(obs_history[-self.config.num_conditioning_frames :])
                obs_np = obs_np.transpose(0, 3, 1, 2)
                obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

                # pass batched observation [1, C, H, W]
                action, policy_hidden = self.actor_critic.get_action(
                    obs_tensor[:, -1],
                    policy_hidden,
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

    def save_checkpoint(self, path: Optional[Union[str, os.PathLike]] = None):
        """Save model checkpoint.

        Args:
            path: Optional path where to write the checkpoint. If `path` is None
                or a bare filename, the file is written into
                `checkpoints/diamond/<filename>`. If `path` contains a directory
                component or is an absolute/relative path, it is used directly.
                When `path` is None, the legacy behavior is preserved and the
                checkpoint is written to `checkpoints/diamond/checkpoint.pt`.
        """
        # Determine output path. Preserve existing behaviour when path is None
        # or a bare filename by writing into checkpoints/diamond.

        default_dir = Path("checkpoints/diamond")
        if path is None:
            default_dir.mkdir(parents=True, exist_ok=True)
            out_path = default_dir / "checkpoint.pt"
        else:
            pathp = Path(path)
            if pathp.parent != Path(""):
                pathp.parent.mkdir(parents=True, exist_ok=True)
                out_path = pathp
            else:
                default_dir.mkdir(parents=True, exist_ok=True)
                out_path = default_dir / pathp
        # Trim replay buffer arrays to current size to avoid saving full-capacity
        # arrays which are wasteful for checkpoints and can blow up memory.
        # Trim and persist replay buffer arrays to separate numpy file(s) to
        # avoid saving large Python objects inside the torch checkpoint. This
        # reduces checkpoint size and avoids unsafe pickle/unpickle usage when
        # restoring Python lists/containers.
        rb_state_trim = None
        replay_file = None
        obs_file = None
        try:
            rb_state = self.replay_buffer.state_dict()
            n = int(self.replay_buffer.size)
            if n > 0:
                rb_state_trim = {
                    "observations": rb_state["observations"][:n].copy(),
                    "next_observations": rb_state["next_observations"][:n].copy(),
                    "actions": rb_state["actions"][:n].copy(),
                    "rewards": rb_state["rewards"][:n].copy(),
                    "dones": rb_state["dones"][:n].copy(),
                    "position": int(self.replay_buffer.position),
                    "size": int(n),
                    "capacity": int(n),
                }
            else:
                rb_state_trim = rb_state
        except Exception:
            rb_state_trim = None

        # Prepare checkpoint (model weights + metadata). We keep hidden states
        # in the torch checkpoint but save large numpy arrays to separate files
        # with a common basename derived from the output path.
        checkpoint = {
            "config": self.config.__dict__,
            "diffusion_model": self.diffusion_model.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
            "diffusion_opt": self.diffusion_opt.state_dict(),
            "reward_opt": self.reward_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            # optional LSTM hidden states saved for reproducible imagination
            "last_policy_hidden": self.last_policy_hidden,
            "last_reward_hidden": self.last_reward_hidden,
        }

        # If we have a trimmed replay buffer, write it to a compressed npz in
        # the same directory as the checkpoint. Also write obs_history_raw as
        # a stacked numpy file if present. Record filenames inside the torch
        # checkpoint for later restoration.
        base = os.path.splitext(str(out_path))[0]
        if rb_state_trim is not None:
            replay_file = base + "_replay.npz"
            # np.savez_compressed will store arrays with the provided keys
            try:
                np.savez_compressed(replay_file, **rb_state_trim)
                checkpoint["replay_buffer_file"] = replay_file
            except Exception:
                # If saving fails, do not embed large Python objects in the
                # torch checkpoint; simply omit the replay buffer files and
                # warn the caller.
                print("Warning: failed to write replay buffer file; skipping embedding")

        if self.obs_history_raw is not None and len(self.obs_history_raw) > 0:
            obs_file = base + "_obs.npy"
            try:
                # Stack into a single array (N, H, W, C)
                obs_arr = np.stack(self.obs_history_raw)
                np.save(obs_file, obs_arr)
                checkpoint["obs_history_file"] = obs_file
            except Exception:
                print("Warning: failed to write obs_history file; skipping embedding")

        torch.save(checkpoint, out_path)

    def load_checkpoint(self, path: Optional[str] = None):
        """Load model checkpoint.

        Args:
            path: Optional path to checkpoint. If None, the default
                `checkpoints/diamond/checkpoint.pt` is loaded. If a bare
                filename is provided, we try `checkpoints/diamond/<filename>`;
                if a path with directory components is provided we use it
                directly.
        """
        # Resolve path similarly to save_checkpoint behaviour
        default_dir = "checkpoints/diamond"
        if path is None:
            fpath = os.path.join(default_dir, "checkpoint.pt")
        else:
            if os.path.exists(path):
                fpath = path
            else:
                alt = os.path.join(default_dir, path)
                if os.path.exists(alt):
                    fpath = alt
                else:
                    raise FileNotFoundError(f"Checkpoint not found at {path} or {alt}")

        # Use full (unsafe) load to restore Python objects (numpy arrays, lists)
        # required for replay buffer and obs history restoration.
        # Load the torch checkpoint (weights + metadata). This file should be
        # safe to load because it contains only tensor state dicts and small
        # metadata fields. Larger numpy arrays may be stored separately and
        # are loaded below when present.
        checkpoint = torch.load(fpath, map_location=self.device, weights_only=False)
        self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.diffusion_opt.load_state_dict(checkpoint["diffusion_opt"])
        self.reward_opt.load_state_dict(checkpoint["reward_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        # restore optional last hidden states (move to configured device)
        if (
            "last_policy_hidden" in checkpoint
            and checkpoint["last_policy_hidden"] is not None
        ):
            h, c = checkpoint["last_policy_hidden"]
            self.last_policy_hidden = (h.to(self.device), c.to(self.device))
        else:
            self.last_policy_hidden = None

        if (
            "last_reward_hidden" in checkpoint
            and checkpoint["last_reward_hidden"] is not None
        ):
            h, c = checkpoint["last_reward_hidden"]
            self.last_reward_hidden = (h.to(self.device), c.to(self.device))
        else:
            self.last_reward_hidden = None

        # restore replay buffer from separate npz file if provided; fall back
        # to embedded dict when necessary.
        if "replay_buffer_file" in checkpoint:
            try:
                replay_file = checkpoint["replay_buffer_file"]
                with np.load(replay_file, allow_pickle=False) as data:
                    rb_state = {k: data[k] for k in data.files}
                self.replay_buffer.load_state_dict(rb_state)
            except Exception:
                print(
                    "Warning: failed to load replay_buffer from file; trying embedded state"
                )
                try:
                    self.replay_buffer.load_state_dict(
                        checkpoint.get("replay_buffer", {})
                    )
                except Exception:
                    print("Warning: failed to load replay_buffer from checkpoint")
        elif "replay_buffer" in checkpoint and checkpoint["replay_buffer"] is not None:
            try:
                self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
            except Exception:
                print("Warning: failed to load replay_buffer from checkpoint")

        # restore obs_history_raw from separate .npy file if present
        if "obs_history_file" in checkpoint:
            try:
                obs_file = checkpoint["obs_history_file"]
                obs_arr = np.load(obs_file, allow_pickle=False)
                self.obs_history_raw = [o for o in obs_arr]
                self.obs_history = [
                    o.astype(np.float32) / 255.0 for o in self.obs_history_raw
                ]
            except Exception:
                print(
                    "Warning: failed to load obs_history from file; trying embedded state"
                )
                try:
                    self.obs_history_raw = checkpoint.get("obs_history_raw", [])
                    self.obs_history = [
                        o.astype(np.float32) / 255.0 for o in self.obs_history_raw
                    ]
                except Exception:
                    print("Warning: failed to load obs_history_raw from checkpoint")
        elif (
            "obs_history_raw" in checkpoint
            and checkpoint["obs_history_raw"] is not None
        ):
            try:
                self.obs_history_raw = checkpoint["obs_history_raw"]
                self.obs_history = [
                    o.astype(np.float32) / 255.0 for o in self.obs_history_raw
                ]
            except Exception:
                print("Warning: failed to load obs_history_raw from checkpoint")


def train_diamond(
    game: str,
    seed: int = 0,
    preset: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train DIAMOND on a specific game."""
    config = DiamondConfig(
        game=game,
        seed=seed,
        preset=preset if preset else None,
        device=device,
    )

    agent = DiamondAgent(config)
    agent.train()


def main(argv: Optional[list[str]] = None) -> None:
    """Parse CLI arguments and train DIAMOND on an Atari game."""
    parser = argparse.ArgumentParser(description="Train DIAMOND on Atari")
    parser.add_argument("--game", type=str, default="Breakout-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preset", type=str, default=None, choices=["small", "medium", "large"]
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    train_diamond(args.game, args.seed, args.preset, args.device)


if __name__ == "__main__":
    main()
