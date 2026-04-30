import argparse
import torch

from world_models.configs.diamond_config import DiamondConfig
from world_models.training.train_diamond import DiamondAgent


def main(game: str, device: str):
    # pick device automatically if user passes 'auto'
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DiamondConfig(game=game, preset="small", device=device)

    # Short smoke-test overrides (fast, not for real training)
    cfg.num_epochs = 3
    cfg.environment_steps_per_epoch = 20
    cfg.training_steps_per_epoch = 10
    cfg.batch_size = 4
    cfg.num_sampling_steps = 2
    cfg.eval_interval = 1
    cfg.save_interval = 1
    cfg.log_interval = 1

    print(f"Smoke-training DIAMOND on {cfg.game} (device={cfg.device})")
    agent = DiamondAgent(cfg)
    agent.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="Breakout-v5")
    parser.add_argument(
        "--device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'"
    )
    args = parser.parse_args()

    main(args.game, args.device)
