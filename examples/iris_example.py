"""
Example script for training an IRIS agent.

IRIS (Imagination with Recurrent State-Space Models) is a world model-based
reinforcement learning agent that uses a transformer-based architecture
for imagination and planning.
"""

import argparse
import logging
import torch
from world_models.models.iris_agent import IRISAgent
from world_models.configs.iris_config import IRISConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train IRIS agent")
    parser.add_argument(
        "--env", type=str, default="cartpole-balance", help="Environment name"
    )
    parser.add_argument(
        "--total-steps", type=int, default=10000, help="Total training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )

    args = parser.parse_args()

    logger.info(f"Training IRIS on {args.env}")

    # Create configuration
    config = IRISConfig()
    config.env = args.env
    config.total_steps = args.total_steps
    config.batch_size = args.batch_size

    if args.device != "auto":
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create agent
    agent = IRISAgent(config, action_size=4, device=device)

    # Example: Load some dummy data and train
    # In a real scenario, you would collect data from the environment
    logger.info(
        "IRIS agent created. For full training, integrate with environment data collection."
    )

    # Example imagination rollout
    initial_frame = torch.randn(2, 3, 64, 64).to(device)
    trajectory = agent.imagine_rollout(initial_frame, horizon=10)
    logger.info(
        f"Imagination rollout completed. Trajectory shapes: {trajectory['frames'].shape}"
    )


if __name__ == "__main__":
    main()
