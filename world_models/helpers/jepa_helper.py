import logging
import sys
from typing import Any, Optional

import torch

import world_models.models.vit as vit
from world_models.utils.jepa_utils import WarmupCosineSchedule, CosineWDSchedule
from world_models.utils.jepa_utils import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device: torch.device,
    r_path: str,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    target_encoder: Optional[torch.nn.Module],
    opt: torch.optim.Optimizer,
    scaler: Optional[Any],
) -> tuple:
    """Load JEPA training state from disk into model and optimizer objects.

    Restores encoder, predictor, optional target encoder, optimizer state,
    and optional AMP scaler, returning the resumed epoch for training restart.
    """
    try:
        checkpoint = torch.load(
            r_path, map_location=torch.device("cpu"), weights_only=True
        )
        epoch = checkpoint["epoch"]

        # -- loading encoder
        pretrained_dict = checkpoint["encoder"]
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

        # -- loading predictor
        pretrained_dict = checkpoint["predictor"]
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint["target_encoder"]
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

        # -- loading optimizer
        opt.load_state_dict(checkpoint["opt"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"loaded optimizers from epoch {epoch}")
        logger.info(f"read-path: {r_path}")
        del checkpoint

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(
    device: torch.device,
    patch_size: int = 16,
    model_name: str = "vit_base",
    crop_size: int = 224,
    pred_depth: int = 6,
    pred_emb_dim: int = 384,
) -> tuple:
    """Initialize JEPA encoder and predictor modules with ViT backbones.

    Applies truncated-normal parameter initialization, moves modules to the
    requested device, and returns `(encoder, predictor)`.
    """
    encoder = vit.__dict__[model_name](img_size=[crop_size], patch_size=patch_size)
    predictor = vit.__dict__["vit_predictor"](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
    )

    def init_weights(m: torch.nn.Module) -> None:
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_opt(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    iterations_per_epoch: int,
    start_lr: float,
    ref_lr: float,
    warmup: float,
    num_epochs: int,
    wd: float = 1e-6,
    final_wd: float = 1e-6,
    final_lr: float = 0.0,
    use_bfloat16: bool = False,
    ipe_scale: float = 1.25,
) -> tuple:
    """Build optimizer, AMP scaler, LR scheduler, and weight-decay scheduler for JEPA.

    Parameters are grouped to exclude bias/norm tensors from weight decay,
    matching typical transformer training best practices.
    """
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]

    logger.info("Using AdamW")
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler("cuda") if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
