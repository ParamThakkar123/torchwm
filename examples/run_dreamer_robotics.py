"""Run DreamerV1 on a Gymnasium Robotics environment.

This small example demonstrates TorchWM's Gymnasium Robotics backend, including
legacy MuJoCo v2/v3 ids that Gymnasium moved into the separate
``gymnasium-robotics`` package.

Install the optional dependency first:
  pip install "torchwm[robotics]"

Usage examples:
  python examples/run_dreamer_robotics.py --env HalfCheetah-v2 --total-steps 2000
  python examples/run_dreamer_robotics.py --list-envs
  python examples/run_dreamer_robotics.py --env FetchReach-v3 --evaluate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger(__name__)


def make_config(args: argparse.Namespace):
    """Create a compact Dreamer config for Robotics smoke runs."""
    from world_models.configs.dreamer_config import DreamerConfig

    cfg = DreamerConfig()
    cfg.env_backend = "robotics"
    cfg.env = args.env
    cfg.gym_render_mode = "rgb_array"
    cfg.total_steps = args.total_steps
    cfg.seed = args.seed
    cfg.no_gpu = args.device == "cpu"
    cfg.algo = "Dreamerv1"

    # Keep defaults small enough for quick examples while preserving the normal
    # Dreamer training loop and environment wrapper stack.
    cfg.seed_steps = args.seed_steps
    cfg.collect_steps = args.collect_steps
    cfg.update_steps = args.update_steps
    cfg.batch_size = args.batch_size
    cfg.image_size = (args.image_size, args.image_size)
    cfg.action_repeat = args.action_repeat
    cfg.time_limit = args.time_limit
    cfg.log_video_freq = -1
    cfg.test_interval = max(1, args.test_interval)
    cfg.test_episodes = args.test_episodes
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DreamerV1 on a Gymnasium Robotics environment"
    )
    parser.add_argument(
        "--env",
        default="HalfCheetah-v2",
        help="Gymnasium Robotics env id, for example HalfCheetah-v2 or FetchReach-v3",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="Print installed Gymnasium Robotics ids and exit",
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
    parser.add_argument(
        "--seed-steps",
        type=int,
        default=500,
        help="Random seed steps for replay warmup",
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
        "--image-size", type=int, default=64, help="Square rendered image size"
    )
    parser.add_argument(
        "--action-repeat", type=int, default=2, help="Action repeat wrapper setting"
    )
    parser.add_argument(
        "--time-limit", type=int, default=1000, help="Episode time limit before repeats"
    )
    parser.add_argument(
        "--test-interval", type=int, default=1000, help="Evaluation interval in steps"
    )
    parser.add_argument(
        "--test-episodes", type=int, default=5, help="Number of evaluation episodes"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("world_models").setLevel(logging.INFO)

    if args.list_envs:
        from world_models.envs.robotics_env import list_gymnasium_robotics_envs

        env_ids = list_gymnasium_robotics_envs()
        if not env_ids:
            raise SystemExit(
                "No Gymnasium Robotics environments found. Install with: "
                "pip install 'torchwm[robotics]'"
            )
        for env_id in env_ids:
            print(env_id)
        return

    from world_models.models.dreamer import DreamerAgent

    cfg = make_config(args)
    LOGGER.info("Running DreamerV1 on Gymnasium Robotics env='%s'", cfg.env)

    agent = DreamerAgent(cfg, logdir=args.logdir)
    if args.evaluate:
        LOGGER.info("Starting evaluation...")
        agent.evaluate()
    else:
        LOGGER.info("Starting training...")
        agent.train(total_steps=args.total_steps)


if __name__ == "__main__":
    main()
