#!/usr/bin/env python3
"""Run inference with a trained Genie checkpoint on TinyWorlds data.

Loads a TinyWorlds sample (or a single image) and generates frames with the
trained Genie model. Saves generated frames as PNGs in `--save_dir`.

Example:
  python scripts/infer_genie_tinyworlds.py \
    --checkpoint checkpoints/genie_sonic_final.pt \
    --data_file /home/azureuser/.cache/tinyworlds/sonic_frames.h5 \
    --dataset SONIC --num_frames 16 --save_dir out/genie_infer
"""

import argparse
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from world_models.training.train_genie import create_genie_trainer
from world_models.datasets import create_tinyworlds_dataloader


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) float tensor to uint8 HxWxC numpy array."""
    arr = t.detach().cpu().numpy()
    # C,H,W -> H,W,C
    arr = np.transpose(arr, (1, 2, 0))
    # clamp and scale
    if arr.max() <= 1.0:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round()
    else:
        arr = np.clip(arr, 0, 255).round()
    return arr.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Infer with Genie checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="SONIC")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--sample_index", type=int, default=0, help="Which dataset sample to use"
    )
    parser.add_argument("--save_dir", type=str, default="out/genie_infer")
    parser.add_argument("--use_maskgit", action="store_true")
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Using device: {device}")

    # Create trainer+model with matching config (use defaults; set num_frames/image_size)
    trainer, model = create_genie_trainer()
    model.to(device)

    # Load checkpoint - support both full trainer checkpoint and model-only state_dict
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        trainer.load_checkpoint(str(ckpt_path))
        print(f"Loaded full trainer checkpoint from {ckpt_path}")
    except Exception:
        # try loading as model state dict only
        state = torch.load(str(ckpt_path), map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            print(f"Loaded model_state_dict from checkpoint {ckpt_path}")
        else:
            model.load_state_dict(state)
            print(f"Loaded state_dict from checkpoint {ckpt_path}")

    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load a sample from TinyWorlds dataset
    dataset, loader = create_tinyworlds_dataloader(
        dataset_name=args.dataset,
        num_frames=args.num_frames,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        download=False,
        data_file=args.data_file,
    )

    if args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {args.sample_index} >= dataset size {len(dataset)}"
        )

    # DataLoader yields batches shaped (B, C, T, H, W)
    # We'll pull the requested sample directly from dataset to avoid batching issues
    sample = dataset[args.sample_index]  # (C, T, H, W)
    # Add batch dim
    sample = sample.unsqueeze(0).to(device)

    # Use the first frame as prompt
    prompt_frame = (
        sample[:, :, 0, :, :].squeeze(2) if sample.dim() == 5 else sample[:, :, 0, :, :]
    )
    # prompt_frame: (B, C, H, W)

    print(f"Prompt frame shape: {prompt_frame.shape}")

    # Generate using model.generate
    with torch.no_grad():
        generated = model.generate(
            prompt_frame,
            num_frames=args.num_frames,
            actions=None,
            use_maskgit=args.use_maskgit,
        )

    # generated: (B, C, num_frames, H, W)
    generated = generated.cpu()
    B = generated.shape[0]

    for b in range(B):
        gen = generated[b]  # (C, T, H, W)
        T = gen.shape[1]
        for t in range(T):
            frame = gen[:, t, :, :]
            img = tensor_to_uint8_img(frame)
            out_path = (
                Path(args.save_dir) / f"sample{args.sample_index}_b{b}_frame{t:03d}.png"
            )
            Image.fromarray(img).save(out_path)

    print(f"Saved generated frames to {args.save_dir}")


if __name__ == "__main__":
    main()
