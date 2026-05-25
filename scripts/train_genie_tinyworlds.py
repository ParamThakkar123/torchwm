#!/usr/bin/env python3
"""Training script for Genie on TinyWorlds HDF5 dataset."""

import argparse
import os
import torch

from world_models.configs.genie_config import GenieSmallConfig
from world_models.training.train_genie import GenieTrainer, create_genie_trainer
from world_models.datasets import create_tinyworlds_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Genie on TinyWorlds dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="SONIC",
        choices=["PICO_DOOM", "PONG", "ZELDA", "POLE_POSITION", "SONIC"],
        help="TinyWorlds dataset to use",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames per video sequence",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Image size for processing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Logging frequency",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1000,
        help="Validation frequency",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for datasets",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to .h5 dataset file (skip download if provided)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--small_config",
        action="store_true",
        help="Use small config for faster training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Using device: {device}")

    if args.small_config:
        config = GenieSmallConfig()
        config.num_frames = args.num_frames
        config.image_size = args.image_size
        config.batch_size = args.batch_size
        config.max_steps = args.max_steps
        config.learning_rate = args.learning_rate
    else:
        config = GenieSmallConfig()
        config.num_frames = args.num_frames
        config.image_size = args.image_size
        config.batch_size = args.batch_size
        config.max_steps = args.max_steps
        config.learning_rate = args.learning_rate

    print(f"Loading {args.dataset} dataset...")
    train_dataset, train_loader = create_tinyworlds_dataloader(
        dataset_name=args.dataset,
        num_frames=args.num_frames,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        cache_dir=args.cache_dir,
        download=not args.data_file,
        data_file=args.data_file,
    )

    print(f"Dataset: {len(train_dataset)} samples, {len(train_loader)} batches")

    print("Creating Genie model and trainer...")
    trainer, model = create_genie_trainer(config=config, device=device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Starting training...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=None,
        num_steps=args.max_steps,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{args.checkpoint_dir}/genie_{args.dataset.lower()}_final.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
