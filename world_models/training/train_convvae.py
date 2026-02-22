"""Training script for Convolutional Variational Autoencoder (ConvVAE).

This module provides functions to train a ConvVAE model on observation data
for world model learning.
"""

import os
from os.path import join, exists

import torch
import torch.utils.data
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from world_models.datasets.wm_dataset import ObservationDataset
from world_models.vision.VAE.ConvVAE import ConvVAE
from world_models.configs.wm_config import WMVAEConfig
from world_models.losses.convae_loss import conv_vae_loss_fn
from world_models.utils.train_utils import EarlyStopping, ReduceLROnPlateau


def save_checkpoint(state, is_best, filename, best_filename):
    """Save model checkpoint.

    Args:
        state: Dictionary containing model state to save.
        is_best: If True, also save as best checkpoint.
        filename: Path to save checkpoint.
        best_filename: Path to save best checkpoint.
    """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def test_epoch(model, test_loader, device, loss_fn):
    """Run one epoch of validation.

    Args:
        model: The VAE model to evaluate.
        test_loader: DataLoader for test/validation data.
        device: Device to run evaluation on.
        loss_fn: Loss function to use.

    Returns:
        float: Average test loss for the epoch.
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data = torch.transpose(data, 1, 3)
            recon, mu, logvar = model(data)
            test_loss += loss_fn(recon, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print("---> Test set loss: {:.4f}".format(test_loss))
    return test_loss


def train_epoch(
    epoch: int,
    model,
    optimizer,
    train_loader,
    device,
    train_dataset,
    loss_fn,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler = None,
):
    """Run one epoch of training.

    Args:
        epoch: Current epoch number.
        model: The VAE model to train.
        optimizer: Optimizer for training.
        train_loader: DataLoader for training data.
        device: Device to run training on.
        train_dataset: Training dataset (used to load next buffer if applicable).
        loss_fn: Loss function to use.
        use_amp: Whether to use automatic mixed precision.
        scaler: GradScaler for mixed precision training.
    """
    model.train()
    train_loss = 0.0
    if hasattr(train_dataset, "load_next_buffer"):
        train_dataset.load_next_buffer()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        data = torch.transpose(data, 1, 3)
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                reconst, mu, logvar = model(data)
                loss = loss_fn(reconst, data, mu, logvar)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            reconst, mu, logvar = model(data)
            loss = loss_fn(reconst, data, mu, logvar)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

        if batch_idx % 20 == 0:
            print(
                f"train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tloss: {loss.item() / len(data):.6f}"
            )

    print(
        "---> Epoch: {} Average Loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def train_convae(config: WMVAEConfig) -> None:
    """Train a Convolutional VAE model.

    This function trains a ConvVAE on observation data using the provided
    configuration. It handles data loading, model initialization, training
    loop, checkpointing, and sample generation.

    Args:
        config: WMVAEConfig object containing all training hyperparameters.

    The training process includes:
        - Loading pretrained VAE if available (unless noreload is True)
        - Training for specified number of epochs
        - Validating after each epoch
        - Learning rate scheduling with ReduceLROnPlateau
        - Early stopping based on validation loss
        - Checkpointing best and current models
        - Generating sample images at specified intervals

    Example:
        >>> config = WMVAEConfig({
        ...     'height': 64,
        ...     'width': 64,
        ...     'latent_size': 32,
        ...     'num_epochs': 100,
        ...     'logdir': 'results',
        ... })
        >>> train_convae(config)
    """
    device = torch.device(config.device if hasattr(config, "device") else "cpu")
    model = ConvVAE(img_channels=3, latent_size=config.latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    earlystopping = EarlyStopping("min", patience=config.early_stopping_patience)

    RED_SIZE = config.height

    vae_dir = join(config.logdir, "vae") if hasattr(config, "logdir") else "vae"
    if not exists(vae_dir):
        os.makedirs(vae_dir)
        os.makedirs(join(vae_dir, "samples"))

    if not config.noreload:
        reload_file = join(vae_dir, "best.tar")
        if exists(reload_file):
            state = torch.load(reload_file)
            print(
                "Reloading model at epoch {}, with test error {}".format(
                    state["epoch"], state["precision"]
                )
            )
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            earlystopping.load_state_dict(state["earlystopping"])

    transform_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((RED_SIZE, RED_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((RED_SIZE, RED_SIZE)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ObservationDataset(
        root=config.data_dir, transform=transform_train, train=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True
    )

    test_dataset = ObservationDataset(
        root=config.data_dir, transform=transform_test, train=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.train_batch_size, shuffle=True
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    cur_best = None

    for ep in range(1, config.num_epochs + 1):
        train_epoch(
            ep,
            model,
            optimizer,
            train_loader,
            device,
            train_dataset,
            conv_vae_loss_fn,
            use_amp=use_amp,
            scaler=scaler,
        )
        test_loss = test_epoch(model, test_loader, device, conv_vae_loss_fn)
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        best_filename = join(vae_dir, "best.tar")
        filename = join(vae_dir, "checkpoint.tar")
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        save_checkpoint(
            {
                "epoch": ep,
                "state_dict": model.state_dict(),
                "precision": test_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "earlystopping": earlystopping.state_dict(),
            },
            is_best,
            filename,
            best_filename,
        )

        if ep % config.sample_interval == 0 and not config.nosamples:
            with torch.no_grad():
                sample = torch.randn(16, config.latent_size).to(device)
                sample = model.decoder(sample).cpu()
                save_image(
                    sample.view(16, 3, RED_SIZE, RED_SIZE),
                    join(vae_dir, "samples", f"sample_epoch_{ep}.png"),
                )

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(ep))
            break

    torch.save(model.state_dict(), join(vae_dir, "convae_final.pth"))
