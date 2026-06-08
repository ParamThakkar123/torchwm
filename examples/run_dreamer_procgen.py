"""Run DreamerV1 on a Procgen environment (small example).

Usage examples:
  python examples/run_dreamer_procgen.py --env coinrun --total-steps 2000
  python examples/run_dreamer_procgen.py --env heist --distribution-mode hard
  python examples/run_dreamer_procgen.py --list-envs
"""

import argparse
import logging

from world_models.configs.dreamer_config import DreamerConfig
from world_models.envs.procgen_env import list_procgen_envs
from world_models.models.dreamer import DreamerAgent


def make_config(args) -> DreamerConfig:
    cfg = DreamerConfig()
    cfg.env_backend = "procgen"
    cfg.env = args.env
    cfg.total_steps = args.total_steps
    cfg.seed = args.seed
    cfg.no_gpu = args.device == "cpu"
    cfg.algo = "Dreamerv1"

    cfg.procgen_distribution_mode = args.distribution_mode
    cfg.procgen_num_levels = args.num_levels
    cfg.procgen_start_level = args.start_level

    cfg.seed_steps = args.seed_steps
    cfg.collect_steps = args.collect_steps
    cfg.update_steps = args.update_steps
    cfg.batch_size = args.batch_size
    cfg.image_size = (64, 64)
    cfg.log_video_freq = -1
    cfg.test_interval = max(1, args.test_interval)
    cfg.test_episodes = args.test_episodes
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run DreamerV1 on Procgen")
    parser.add_argument(
        "--env", default="coinrun", help="Procgen game name (e.g. coinrun, heist)"
    )
    parser.add_argument(
        "--distribution-mode",
        default="easy",
        help="Procgen distribution mode (e.g. easy, hard, extreme)",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=0,
        help="Number of training levels; 0 uses unlimited levels",
    )
    parser.add_argument(
        "--start-level",
        type=int,
        default=None,
        help="First Procgen level; defaults to the configured seed",
    )
    parser.add_argument(
        "--total-steps", type=int, default=5000, help="Total training steps"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation instead of training"
    )
    parser.add_argument("--logdir", type=str, default=None, help="Optional logdir")
    parser.add_argument("--list-envs", action="store_true", help="List Procgen games")
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
    if args.list_envs:
        for env_name in list_procgen_envs():
            print(env_name)
        return

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("world_models").setLevel(logging.INFO)

    cfg = make_config(args)
    logging.info(
        "Running DreamerV1 on procgen env='%s' (distribution=%s, levels=%s)",
        cfg.env,
        cfg.procgen_distribution_mode,
        cfg.procgen_num_levels,
    )

    agent = DreamerAgent(cfg, logdir=args.logdir)
    if args.evaluate:
        logging.info("Starting evaluation...")
        agent.evaluate()
    else:
        logging.info("Starting training...")
        agent.train(total_steps=args.total_steps)


if __name__ == "__main__":
    main()
