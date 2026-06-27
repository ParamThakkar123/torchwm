#!/usr/bin/env python3
"""Train DIAMOND on an Atari game to convergence (Atari 100k protocol).

Usage:
    python scripts/train_diamond.py
    python scripts/train_diamond.py game=Pong-v5 preset=small
"""

from world_models.configs.diamond_config import DiamondConfig
from world_models.training.train_diamond import train_diamond


def main():
    from omegaconf import OmegaConf

    cli_cfg = OmegaConf.from_cli()

    game = cli_cfg.get("game", "Breakout-v5")
    preset = cli_cfg.get("preset", "medium")
    seed = cli_cfg.get("seed", 0)
    device = cli_cfg.get("device", None)
    num_epochs = cli_cfg.get("num_epochs", None)

    cfg = DiamondConfig(
        game=game,
        preset=preset,
        seed=seed,
        device=device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
    )
    if num_epochs is not None:
        cfg.num_epochs = int(num_epochs)
    # Windows compatibility: multiprocessing spawn with >0 workers crashes
    cfg.data_loader_num_workers = 0
    cfg.persistent_workers = False
    cfg.pin_memory = False

    print(f"Game:        {cfg.game}")
    print(f"Preset:      {cfg.preset or 'default'}")
    print(f"Seed:        {cfg.seed}")
    print(f"Device:      {cfg.device}")
    print(f"Num epochs:  {cfg.num_epochs}")
    print(f"Obs size:    {cfg.obs_size}")
    print(f"Batch size:  {cfg.batch_size}")
    print(f"Eval every:  {cfg.eval_interval} epochs")
    print(f"Save every:  {cfg.save_interval} epochs")

    train_diamond(config=cfg)


if __name__ == "__main__":
    main()
