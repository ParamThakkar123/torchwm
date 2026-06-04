"""Run DreamerV1 on a Brax environment (small example).

This lightweight script demonstrates how to run the DreamerAgent from
torchwm (world_models) against a Brax environment. It is intentionally
minimal so you can adapt it for experiments or quick smoke tests.

Usage examples:
  python examples/run_dreamer_brax.py --env ant --total-steps 2000
  python examples/run_dreamer_brax.py --env walker --evaluate
"""

import argparse
import logging

from world_models.models.dreamer import DreamerAgent
from world_models.configs.dreamer_config import DreamerConfig


def make_config(args) -> DreamerConfig:
    cfg = DreamerConfig()
    cfg.env_backend = "brax"
    cfg.env = args.env
    cfg.total_steps = args.total_steps
    cfg.seed = args.seed
    cfg.no_gpu = args.device == "cpu"
    cfg.brax_jit = args.brax_jit
    cfg.brax_auto_reset = args.brax_auto_reset
    # Use Dreamer v1 explicitly
    cfg.algo = "Dreamerv1"
    # Shorter defaults for example runs
    cfg.seed_steps = args.seed_steps
    cfg.collect_steps = args.collect_steps
    cfg.update_steps = args.update_steps
    cfg.batch_size = args.batch_size
    cfg.image_size = (64, 64)
    # Disable heavy logging/video by default for quick runs
    cfg.log_video_freq = -1
    cfg.test_interval = max(1, args.test_interval)
    cfg.test_episodes = args.test_episodes
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run DreamerV1 on Brax")
    parser.add_argument(
        "--env", default="ant", help="Brax env name (e.g. ant, walker, cheetah)"
    )
    parser.add_argument(
        "--total-steps", type=int, default=5000, help="Total training steps"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--brax-jit",
        action="store_true",
        help="Enable JAX jit for Brax (default: False)",
    )
    parser.add_argument(
        "--brax-auto-reset",
        action="store_true",
        help="Enable Brax auto_reset (default: False)",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation instead of training"
    )
    parser.add_argument("--logdir", type=str, default=None, help="Optional logdir")
    parser.add_argument(
        "--seed-steps",
        type=int,
        default=500,
        help="Random seed steps to populate buffer",
    )
    parser.add_argument(
        "--collect-steps", type=int, default=200, help="Steps to collect per iteration"
    )
    parser.add_argument(
        "--update-steps", type=int, default=10, help="Gradient update batches per loop"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--test-interval", type=int, default=1000, help="Evaluation interval (steps)"
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=5,
        help="Number of eval episodes when testing",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("world_models").setLevel(logging.INFO)

    cfg = make_config(args)
    logging.info(f"Running DreamerV1 on brax env='{cfg.env}' (jit={cfg.brax_jit})")

    agent = DreamerAgent(cfg, logdir=args.logdir)

    if args.evaluate:
        logging.info("Starting evaluation...")
        agent.evaluate()
    else:
        logging.info("Starting training...")
        agent.train(total_steps=args.total_steps)


if __name__ == "__main__":
    main()
