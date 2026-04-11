"""
Example script for training a Dreamer agent on a Gym environment.

This demonstrates how to use the DreamerAgent class for end-to-end training
of a world model-based reinforcement learning agent.
"""

import argparse
import logging
from world_models.models.dreamer import DreamerAgent
from world_models.configs.dreamer_config import DreamerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Dreamer agent")
    parser.add_argument(
        "--env",
        type=str,
        default="cartpole_balance",
        help="Environment name (DMC or Gym format)",
    )
    parser.add_argument(
        "--total-steps", type=int, default=10000, help="Total training steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--logdir", type=str, default=None, help="Logging directory (optional)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )

    args = parser.parse_args()

    logger.info(f"Training Dreamer on {args.env}")
    logger.info(f"Total steps: {args.total_steps}")

    # Create configuration
    config = DreamerConfig()
    config.env = args.env
    config.total_steps = args.total_steps
    config.seed = args.seed

    if args.device != "auto":
        config.no_gpu = args.device == "cpu"

    # Create and train agent
    agent = DreamerAgent(config, logdir=args.logdir)
    agent.train(total_steps=args.total_steps)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
