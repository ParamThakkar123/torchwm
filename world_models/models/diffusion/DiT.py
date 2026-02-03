import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from world_models.configs.dit_config import DiTConfig as Config
from world_models.layers.AdaLNNorm import AdaLNNormalization
from world_models.blocks.mhsa import MultiHeadSelfAttention
from world_models.models.diffusion.DDPM import DDPM
from world_models.datasets.cifar10 import make_cifar10
from world_models.datasets.imagenet1k import make_imagenet1k, make_imagefolder
from torchvision.transforms import RandomHorizontalFlip, Compose, ToTensor
from world_models.transforms.transforms import make_transforms
import time
from torchvision.utils import save_image
import os

cfg = Config()


def sinusoidal_time_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(10000.0), half, device=timesteps.device)
    )
    args = timesteps.float().unsqueeze(1) / cfg.TIMESTEPS * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.pos
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, out_channels):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.ConvTranspose2d(
            embed_dim, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        h = w = self.img_size // self.patch_size
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, drop, t_dim):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = AdaLNNormalization(d_model, t_dim)
        self.norm2 = AdaLNNormalization(d_model, t_dim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(mlp_ratio * d_model), d_model),
            nn.Dropout(drop),
        )

    def forward(self, x, t_emb):
        h = self.norm1(x, t_emb)
        attn_out = self.attn(h)
        x = x + attn_out
        h = self.norm2(x, t_emb)
        ff_out = self.ff(h)
        x = x + ff_out
        return x


class DiT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        d_model,
        depth,
        heads,
        drop=0.0,
        t_dim=256,
    ):
        super(DiT, self).__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.GELU(),
            nn.Linear(t_dim, t_dim),
        )
        self.patchify = PatchEmbed(img_size, patch_size, in_channels, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, heads, mlp_ratio=4.0, drop=drop, t_dim=t_dim)
                for _ in range(depth)
            ]
        )

        self.unpatchify = PatchUnEmbed(img_size, patch_size, d_model, in_channels)
        self.out = nn.Identity()

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)

        x = self.patchify(x)
        for block in self.transformer_blocks:
            x = block(x, t_emb)
        x = self.unpatchify(x)
        x = self.out(x)
        return x

    @classmethod
    def train(
        cls,
        epochs,
        dataset,
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
        workdir="./dit_demo",
        root_path="./data",
        image_folder=None,
        crop_size=224,
        download=True,
        copy_data=False,
        subset_file=None,
        val_split=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dataset.lower() == "cifar10":
            transform = Compose([RandomHorizontalFlip(), ToTensor()])
        else:
            transform = make_transforms(
                crop_size=crop_size,
                crop_scale=(0.3, 1.0),
                color_jitter=0.5,
                horizontal_flip=True,
                color_distortion=True,
                gaussian_blur=True,
                normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )

        if dataset.lower() == "cifar10":
            _, train_loader, _ = make_cifar10(
                transform=transform,
                batch_size=batch_size,
                collator=None,
                pin_mem=True,
                num_workers=4,
                world_size=1,
                rank=0,
                root_path=root_path,
                drop_last=True,
                train=True,
                download=download,
            )
        elif dataset.lower() == "imagenet":
            _, train_loader, _ = make_imagenet1k(
                transform=transform,
                batch_size=batch_size,
                collator=None,
                pin_mem=True,
                num_workers=4,
                world_size=1,
                rank=0,
                root_path=root_path,
                image_folder=image_folder,
                training=True,
                copy_data=copy_data,
                drop_last=True,
                subset_file=subset_file,
            )
        elif dataset.lower() == "imagefolder":
            _, train_loader, _ = make_imagefolder(
                transform=transform,
                batch_size=batch_size,
                collator=None,
                pin_mem=True,
                num_workers=4,
                world_size=1,
                rank=0,
                root_path=root_path,
                image_folder=image_folder,
                drop_last=True,
                val_split=val_split,
            )
        else:
            raise ValueError(
                f"Unsupported dataset: {dataset}. Supported: cifar10, imagenet, imagefolder"
            )

        ddpm = DDPM(
            timesteps=timesteps, beta_start=beta_start, beta_end=beta_end, device=device
        )

        model = cls(
            img_size=img_size,
            patch_size=patch,
            in_channels=channels,
            d_model=width,
            depth=depth,
            heads=heads,
            drop=drop,
            t_dim=256,
        ).to(device)

        def param_count(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model Parameters: {param_count(model)/1e6:.2f}M")

        ema_model = None
        if ema:
            import copy

            ema_model = copy.deepcopy(model).to(device).eval()
            for p in ema_model.parameters():
                p.requires_grad = False

        def ema_update(m, ema_m, decay=ema_decay):
            with torch.no_grad():
                for p, q in zip(m.parameters(), ema_m.parameters()):
                    q.data.mul_(decay).add_(p.data, alpha=1 - decay)

        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        global_step = 0
        model.train()

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            for imgs, _ in train_loader:
                imgs = imgs.to(device)
                b = imgs.size(0)
                t = torch.randint(0, timesteps, (b,), device=device).long()
                noise = torch.randn_like(imgs)
                x_t = ddpm.q_sample(imgs, t, noise)

                pred = model(x_t, t)
                loss = F.mse_loss(pred, noise)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if ema_model is not None:
                    ema_update(model, ema_model)

                if global_step % 100 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Epoch [{epoch}/{epochs}] Step [{global_step}] Loss: {loss.item():.4f} Time Elapsed: {elapsed/60:.2f} min"
                    )
                    start_time = time.time()

                global_step += 1
        print("Training Complete.")

        os.makedirs(workdir, exist_ok=True)

        model_to_save = ema_model if ema_model is not None else model
        torch.save(model_to_save.state_dict(), f"{workdir}/dit_model.pth")
        print(f"Model saved to {workdir}/dit_model.pth")

        # Generate new Images
        model_to_sample = ema_model if ema_model is not None else model
        model_to_sample.eval()
        with torch.no_grad():
            samples = ddpm.sample(
                model_to_sample, n=16, img_size=img_size, channels=channels
            )
            os.makedirs(workdir, exist_ok=True)
            save_image((samples + 1) / 2, f"{workdir}/generated_samples.png", nrow=4)
            print(f"Generated samples saved to {workdir}/generated_samples.png")
