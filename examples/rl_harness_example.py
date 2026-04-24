"""
Example script for running the RL harness with vectorized simulators.
This demonstrates the implementation of vectorized simulator workers and RL training harness.
"""

import argparse
import logging
from functools import partial
from world_models.training.rl_harness import PPOTrainer
from world_models.envs.vector_env import TorchVectorizedEnv
from world_models.envs.gym_env import make_gym_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run RL harness with vectorized environments"
    )
    parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Gym environment ID to use"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of worker processes"
    )
    parser.add_argument(
        "--envs-per-worker",
        type=int,
        default=4,
        help="Number of environments per worker",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000,
        help="Total timesteps to train for",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run training on (cpu/cuda)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    logger.info(f"Starting RL harness with env: {args.env}")
    logger.info(f"Workers: {args.num_workers}, Envs per worker: {args.envs_per_worker}")
    logger.info(f"Total envs: {args.num_workers * args.envs_per_worker}")
    logger.info(f"Total timesteps: {args.total_timesteps}")

    # Create vectorized environment
    env_factory = partial(make_gym_env, env=args.env, size=(64, 64))
    vec_env = TorchVectorizedEnv(
        env_factory=env_factory,
        num_workers=args.num_workers,
        envs_per_worker=args.envs_per_worker,
        seed=args.seed,
    )

    # Create and run trainer
    trainer = PPOTrainer(vec_env, device=args.device)
    trainer.train(total_timesteps=args.total_timesteps)

    logger.info("Training completed. Cleaning up...")
    vec_env.close()


if __name__ == "__main__":
    main()
