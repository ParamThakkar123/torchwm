from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM(nn.Module):
    """Utility module implementing forward and reverse DDPM diffusion steps.

    Precomputes diffusion schedule terms and exposes helpers for noising
    training inputs (`q_sample`) and iterative denoising sampling (`sample`).
    """

    def __init__(self, timesteps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.betas: torch.Tensor = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - self.betas
        self.alphas: torch.Tensor = alphas
        self.alphas_cumprod: torch.Tensor = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev: torch.Tensor = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )

        self.register_buffer("betas", self.betas)
        self.register_buffer("alphas", self.alphas)
        self.register_buffer("alphas_cumprod", self.alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", self.alphas_cumprod_prev)
        self.sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(self.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", self.sqrt_alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", self.sqrt_one_minus_alphas_cumprod
        )
        self.posterior_variance: torch.Tensor = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", self.posterior_variance)

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return s1 * x_start + s2 * noise

    def p_sample(
        self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # Predict noise
        eps = model(x_t, t)
        # Compute x0_hat
        a_t = self.alphas[t].view(-1, 1, 1, 1)
        ac_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_hat = (x_t - sqrt_one_minus_ac * eps) / torch.sqrt(ac_t)

        # Compute Posterior mean
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        ac_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        coef1 = torch.sqrt(ac_prev) * beta_t / (1.0 - ac_t)
        coef2 = torch.sqrt(a_t) * (1.0 - ac_prev) / (1.0 - ac_t)
        mean = coef1 * x0_hat + coef2 * x_t

        # Add noise except for t == 0
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(
        self, model: nn.Module, n: int, img_size: int, channels: int
    ) -> torch.Tensor:
        x = torch.randn(n, channels, img_size, img_size).to(
            next(model.parameters()).device
        )
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, dtype=torch.long).to(x.device)
            x = self.p_sample(model, x, t)
        return x.clamp(-1.0, 1.0)
