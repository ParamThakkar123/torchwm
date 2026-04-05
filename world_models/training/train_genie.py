import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class GenieConfig:
    """Configuration for Genie training."""

    num_frames: int = 16
    image_size: int = 64
    in_channels: int = 3

    tokenizer_vocab_size: int = 1024
    tokenizer_embedding_dim: int = 32
    tokenizer_encoder_dim: int = 512
    tokenizer_decoder_dim: int = 1024
    tokenizer_encoder_depth: int = 12
    tokenizer_decoder_depth: int = 20

    action_vocab_size: int = 8
    action_embedding_dim: int = 32
    action_encoder_dim: int = 1024
    action_encoder_depth: int = 20

    dynamics_dim: int = 512
    dynamics_depth: int = 8
    dynamics_num_heads: int = 8

    batch_size: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 5000
    max_steps: int = 125000

    mask_prob_min: float = 0.5
    mask_prob_max: float = 1.0

    sample_temperature: float = 2.0
    maskgit_steps: int = 25


class VideoDataset(Dataset):
    """Dataset for video data."""

    def __init__(self, video_paths: list, num_frames: int = 16, image_size: int = 64):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        raise NotImplementedError("Implement loading video from paths")


class GenieTrainer:
    """Trainer for Genie model."""

    def __init__(
        self,
        model: nn.Module,
        config: GenieConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = self._create_scheduler()

        self.global_step = 0

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup and cosine decay."""
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single training step.

        Args:
            batch: (B, C, T, H, W) video batch

        Returns:
            Dictionary of losses
        """
        self.model.train()

        B, C, T, H, W = batch.shape
        mask_prob = (
            torch.rand(1).item()
            * (self.config.mask_prob_max - self.config.mask_prob_min)
            + self.config.mask_prob_min
        )

        outputs = self.model(batch, mask_prob=mask_prob)

        recon_loss = outputs["tokenizer_loss"].get("recon_loss", 0.0)
        vq_loss = outputs["tokenizer_loss"].get("vq_loss", 0.0)

        if "dynamics_logits" in outputs and outputs["dynamics_logits"] is not None:
            target_tokens = outputs["video_indices"][:, 1:, :]
            dynamics_logits = outputs["dynamics_logits"]

            B, T_pred, N, V = dynamics_logits.shape
            target_tokens_flat = target_tokens.reshape(B * T_pred * N)
            logits_flat = dynamics_logits.reshape(B * T_pred * N, V)

            dynamics_loss = F.cross_entropy(logits_flat, target_tokens_flat)
        else:
            dynamics_loss = torch.tensor(0.0, device=self.device)

        total_loss = recon_loss + vq_loss + dynamics_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item()
            if isinstance(recon_loss, torch.Tensor)
            else recon_loss,
            "vq_loss": vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss,
            "dynamics_loss": dynamics_loss.item()
            if isinstance(dynamics_loss, torch.Tensor)
            else dynamics_loss,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

    def validate(self, val_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            val_batch: (B, C, T, H, W) validation video batch

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(val_batch, mask_prob=0.0)

            recon_loss = outputs["tokenizer_loss"].get("recon_loss", 0.0)

            return {
                "val_recon_loss": recon_loss.item()
                if isinstance(recon_loss, torch.Tensor)
                else recon_loss,
            }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_steps: Optional[int] = None,
        log_interval: int = 100,
        val_interval: int = 1000,
    ):
        """Full training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_steps: Number of training steps (uses config.max_steps if None)
            log_interval: Logging frequency
            val_interval: Validation frequency
        """
        if num_steps is None:
            num_steps = self.config.max_steps

        train_iter = iter(train_dataloader)

        while self.global_step < num_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            batch = batch.to(self.device)

            losses = self.train_step(batch)

            if self.global_step % log_interval == 0:
                print(
                    f"Step {self.global_step}/{num_steps} | "
                    f"Loss: {losses['total_loss']:.4f} | "
                    f"Recon: {losses['recon_loss']:.4f} | "
                    f"VQ: {losses['vq_loss']:.4f} | "
                    f"Dynamics: {losses['dynamics_loss']:.4f} | "
                    f"LR: {losses['learning_rate']:.6f}"
                )

            if val_dataloader is not None and self.global_step % val_interval == 0:
                val_iter = iter(val_dataloader)
                try:
                    val_batch = next(val_iter)
                    val_batch = val_batch.to(self.device)
                    val_metrics = self.validate(val_batch)
                    print(f"Validation: {val_metrics}")
                except StopIteration:
                    pass

        print("Training complete!")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]


def create_genie_trainer(
    config: Optional[GenieConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[GenieTrainer, nn.Module]:
    """Factory function to create Genie trainer and model."""
    if config is None:
        config = GenieConfig()

    from world_models.models.genie import Genie

    model = Genie(
        num_frames=config.num_frames,
        image_size=config.image_size,
        tokenizer_vocab_size=config.tokenizer_vocab_size,
        tokenizer_embedding_dim=config.tokenizer_embedding_dim,
        action_vocab_size=config.action_vocab_size,
        action_embedding_dim=config.action_embedding_dim,
        dynamics_dim=config.dynamics_dim,
        dynamics_depth=config.dynamics_depth,
        dynamics_num_heads=config.dynamics_num_heads,
        encoder_depth=config.tokenizer_encoder_depth,
        decoder_depth=config.tokenizer_decoder_depth,
        latent_action_depth=config.action_encoder_depth,
    )

    trainer = GenieTrainer(model, config, device)

    return trainer, model
