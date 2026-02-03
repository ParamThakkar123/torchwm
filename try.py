from world_models.models.diffusion.DiT import DiT

# Train DiT on CIFAR10
DiT.train(
    epochs=10,
    dataset="cifar10",
    batch_size=128,
    lr=2e-4,
    img_size=32,
    channels=3,
    patch=4,
    width=384,
    depth=6,
    heads=6,
    drop=0.1,
    timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    ema=True,
    ema_decay=0.999,
    workdir="./dit_cifar10_demo",
    root_path="./data",
    download=True,
)
