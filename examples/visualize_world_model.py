"""
World Model Visualization Script

This script loads a trained world model and visualizes its "imagination" -
the trajectories it generates internally. You can watch the model dream
about what the environment looks like.

==============================================================================
USAGE
==============================================================================

Basic - Generate pure imagination (model dreaming):
    python examples/visualize_world_model.py

With trained checkpoint:
    python examples/visualize_world_model.py --checkpoint checkpoints/diamond_pybullet/checkpoint_0.pt

Custom number of steps:
    python examples/visualize_world_model.py --num_steps 200 --output my_dream.mp4

Compare real environment vs world model imagination:
    python examples/visualize_world_model.py --compare --num_steps 100

Use GPU if available:
    python examples/visualize_world_model.py --device cuda

==============================================================================
ARGUMENTS
==============================================================================

    --checkpoint  : Path to model checkpoint (default: checkpoints/diamond_pybullet/checkpoint_0.pt)
    --num_steps   : Number of steps to visualize (default: 100)
    --output      : Output video path (default: imagination.mp4)
    --compare     : Show side-by-side real vs imagined comparison
    --device      : Device to run on - cuda or cpu (default: auto-detect)

==============================================================================
WHAT IT SHOWS
==============================================================================

The visualization displays what the world model "imagines" the environment to be:

1. **Imagination only (default)**:
   - Starts with random conditioning frames
   - Uses diffusion model to generate observations
   - Uses reward model to predict rewards/dones
   - Creates a video of the model's "dreams"

2. **Comparison mode (--compare)**:
   - Left side: Real environment
   - Right side: World model's prediction
   - Helps verify how well the model learns the environment

This is useful for:
- Understanding what the world model has learned
- Debugging training issues
- Visualizing the model's internal representation
- Comparing real vs imagined trajectories

==============================================================================
"""

import torch
import numpy as np
import cv2
import os
from pathlib import Path

from world_models.configs.diamond_config import DiamondConfig
from world_models.models.diffusion.diamond_diffusion import (
    DiffusionUNet,
    EDMPreconditioner,
    EulerSampler,
)
from world_models.models.diffusion.reward_termination import RewardTerminationModel
from world_models.models.diffusion.actor_critic import ActorCriticNetwork


def create_video_writer(output_path, fps=30):
    """Create OpenCV video writer."""
    return cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 256)
    )


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization."""
    if tensor.device.type == "cuda":
        tensor = tensor.cpu()
    arr = tensor.detach().numpy()
    if arr.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        arr = arr.transpose(1, 2, 0)
    return (arr * 255).astype(np.uint8)


def visualize_imagination(
    diffusion_model,
    reward_model,
    sampler,
    config,
    device,
    num_steps=100,
    save_path="imagination.mp4",
    action_dim=11,
):
    """
    Generate and visualize imagined trajectories from the world model.
    """
    print(f"Generating imagination visualization...")

    B = 1  # Single trajectory
    num_conditioning = config.num_conditioning_frames

    # Start with random observations
    obs_history = torch.randn(
        B, num_conditioning, 3, config.obs_size, config.obs_size, device=device
    )
    actions = torch.randint(0, action_dim, (B, num_conditioning), device=device)

    hidden_state = reward_model.init_hidden(B, device)

    video_writer = create_video_writer(save_path, fps=15)

    for step in range(num_steps):
        # Sample next observation using diffusion model
        next_obs = sampler.sample(
            model=diffusion_model,
            shape=(B, 3, config.obs_size, config.obs_size),
            device=device,
            obs_history=obs_history,
            actions=actions,
        )

        # Get reward prediction
        reward, done, hidden_state = reward_model.predict(
            obs=next_obs,
            actions=actions[:, -1],
            hidden_state=hidden_state,
        )

        # Visualize the imagined observation
        frame = tensor_to_numpy(next_obs[0])

        # Resize to 256x256 for better visibility
        frame = cv2.resize(frame, (256, 256))

        # Add text overlay
        reward_val = reward.item() if isinstance(reward, torch.Tensor) else reward
        cv2.putText(
            frame,
            f"Step: {step}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Reward: {reward_val:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        video_writer.write(frame)

        # Update history for next step
        next_obs_seq = next_obs.unsqueeze(1)  # (B, 1, C, H, W)
        obs_history = torch.cat([obs_history[:, 1:], next_obs_seq], dim=1)

        new_action = torch.randint(0, action_dim, (B, 1), device=device)
        actions = torch.cat([actions[:, 1:], new_action], dim=1)

        if step % 10 == 0:
            print(f"  Generated step {step}/{num_steps}")

    video_writer.release()
    print(f"Saved imagination video to: {save_path}")

    return save_path


def visualize_comparison(
    real_env,
    diffusion_model,
    reward_model,
    sampler,
    config,
    device,
    num_steps=100,
    save_path="comparison.mp4",
    action_dim=11,
):
    """
    Compare real environment vs world model imagination side by side.
    """
    print(f"Generating comparison visualization...")

    B = 1
    num_conditioning = config.num_conditioning_frames

    # Reset real environment
    real_obs, _ = real_env.reset()
    real_obs = real_obs.astype(np.float32) / 255.0

    # Initialize observation history from real environment
    obs_history = (
        torch.from_numpy(np.stack([real_obs] * num_conditioning))
        .unsqueeze(0)
        .permute(0, 1, 3, 4, 2)
        .to(device)
    )  # (B, num_cond, H, W, C)
    obs_history = obs_history.permute(0, 1, 4, 2, 3)  # (B, num_cond, C, H, W)

    actions = torch.randint(0, action_dim, (B, num_conditioning), device=device)

    hidden_state = reward_model.init_hidden(B, device)

    video_writer = create_video_writer(save_path, fps=15)

    for step in range(num_steps):
        # Get real observation
        real_frame = cv2.resize(real_obs, (256, 256))

        # Sample imagined observation
        imagined_obs = sampler.sample(
            model=diffusion_model,
            shape=(B, 3, config.obs_size, config.obs_size),
            device=device,
            obs_history=obs_history,
            actions=actions,
        )

        imagined_frame = tensor_to_numpy(imagined_obs[0])
        imagined_frame = cv2.resize(imagined_frame, (256, 256))

        # Create side-by-side comparison
        combined = np.concatenate([real_frame, imagined_frame], axis=1)

        # Add labels
        cv2.putText(
            combined,
            "Real Environment",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            combined,
            "World Model Imagination",
            (266, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        video_writer.write(combined)

        # Take real action in real environment
        action = np.random.randint(0, action_dim)
        real_obs, reward, done, _ = real_env.step(action)
        real_obs = real_obs.astype(np.float32) / 255.0

        # Get reward from world model
        reward_pred, done_pred, hidden_state = reward_model.predict(
            obs=imagined_obs,
            actions=actions[:, -1],
            hidden_state=hidden_state,
        )

        # Update histories
        real_obs_tensor = (
            torch.from_numpy(real_obs).permute(2, 0, 1).unsqueeze(0).to(device)
        )
        obs_history = torch.cat([obs_history[:, 1:], real_obs_tensor], dim=1)

        new_action = torch.tensor([[action]], device=device)
        actions = torch.cat([actions[:, 1:], new_action], dim=1)

        if step % 10 == 0:
            print(f"  Step {step}/{num_steps}")

        if done:
            real_obs, _ = real_env.reset()
            real_obs = real_obs.astype(np.float32) / 255.0
            obs_history = (
                torch.from_numpy(np.stack([real_obs] * num_conditioning))
                .unsqueeze(0)
                .permute(0, 1, 3, 4, 2)
                .to(device)
                .permute(0, 1, 4, 2, 3)
            )

    video_writer.release()
    print(f"Saved comparison video to: {save_path}")

    return save_path


def load_trained_models(checkpoint_path, config, action_dim, device):
    """Load trained DIAMOND models from checkpoint."""
    print(f"Loading models from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create models
    base_ch = 32

    diffusion_model = DiffusionUNet(
        obs_channels=3,
        num_conditioning_frames=config.num_conditioning_frames,
        base_channels=base_ch,
        channel_multipliers=tuple(config.diffusion_channels),
        num_res_blocks=config.diffusion_res_blocks,
        cond_dim=config.diffusion_cond_dim,
        action_dim=action_dim,
    ).to(device)

    reward_model = RewardTerminationModel(
        obs_channels=3,
        action_dim=action_dim,
        channels=tuple(config.reward_channels),
        lstm_dim=config.reward_lstm_dim,
        cond_dim=config.reward_cond_dim,
    ).to(device)

    edm_precond = EDMPreconditioner(
        sigma_data=config.sigma_data,
        p_mean=config.p_mean,
        p_std=config.p_std,
    )

    sampler = EulerSampler(
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
        num_steps=config.num_sampling_steps,
    )

    # Load weights
    diffusion_model.load_state_dict(checkpoint["diffusion_model"])
    reward_model.load_state_dict(checkpoint["reward_model"])

    diffusion_model.eval()
    reward_model.eval()

    print("Models loaded successfully!")

    return diffusion_model, reward_model, edm_precond, sampler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize world model imagination")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/diamond_pybullet/checkpoint_0.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of steps to visualize"
    )
    parser.add_argument(
        "--output", type=str, default="imagination.mp4", help="Output video path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Create config
    config = DiamondConfig(
        game="PyBullet-Box",
        preset=None,
        device=args.device,
        obs_size=64,
    )
    config.diffusion_channels = [1, 1, 1, 1]
    config.diffusion_res_blocks = 2
    config.diffusion_cond_dim = 128
    config.reward_channels = [8, 16, 32, 32]
    config.reward_lstm_dim = 256
    config.num_conditioning_frames = 4
    config.imagination_horizon = 15
    config.sigma_data = 0.5
    config.p_mean = -0.5
    config.p_std = 1.0
    config.sigma_min = 0.002
    config.sigma_max = 80.0
    config.rho = 7.0
    config.num_sampling_steps = 40

    action_dim = 11

    # Load models
    if os.path.exists(args.checkpoint):
        diffusion_model, reward_model, edm_precond, sampler = load_trained_models(
            args.checkpoint, config, action_dim, device
        )
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Creating new untrained models for visualization...")

        diffusion_model = DiffusionUNet(
            obs_channels=3,
            num_conditioning_frames=config.num_conditioning_frames,
            base_channels=32,
            channel_multipliers=(1, 1, 1, 1),
            num_res_blocks=2,
            cond_dim=128,
            action_dim=action_dim,
        ).to(device)

        reward_model = RewardTerminationModel(
            obs_channels=3,
            action_dim=action_dim,
            channels=(8, 16, 32, 32),
            lstm_dim=256,
            cond_dim=128,
        ).to(device)

        edm_precond = EDMPreconditioner(
            sigma_data=config.sigma_data,
            p_mean=config.p_mean,
            p_std=config.p_std,
        )

        sampler = EulerSampler(
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            rho=config.rho,
            num_steps=config.num_sampling_steps,
        )

    # Generate visualization
    visualize_imagination(
        diffusion_model=diffusion_model,
        reward_model=reward_model,
        sampler=sampler,
        config=config,
        device=device,
        num_steps=args.num_steps,
        save_path=args.output,
        action_dim=action_dim,
    )

    print(f"\nDone! Video saved to: {args.output}")
    print(f"To view it, open {args.output} in a video player.")
