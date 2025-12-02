import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import TanhTransform


class SampleDist:
    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        samples = self._dist.rsample((self._samples,))
        return samples.mean(0)

    def mode(self):
        samples = self._dist.rsample((self._samples,))
        logprob = self._dist.log_prob(samples)
        idx = torch.argmax(logprob, dim=0)
        return samples[idx, torch.arange(samples.size(1))]

    def entropy(self):
        samples = self._dist.rsample((self._samples,))
        logprob = self._dist.log_prob(samples)
        return -logprob.mean(0)


class ActorModel(nn.Module):
    def __init__(
        self,
        action_size,
        belief_size,
        state_size,
        hidden_size,
        mean_scale=5,
        min_std=1e-4,
        init_std=0.0,
        activation_function="elu",
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.net = nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
        )
        self.mean_layer = nn.Linear(hidden_size, action_size)
        self.std_layer = nn.Linear(hidden_size, action_size)
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale

    def forward(self, belief, state, deterministic=False, with_logprob=False):

        x = torch.cat([belief, state], dim=-1)
        x = self.net(x)

        mean = self.mean_layer(x) / self.mean_scale

        std_logits = self.std_layer(x)
        std = F.softplus(std_logits + self.init_std) + self.min_std

        base_dist = Normal(mean, std)
        transform = [TanhTransform()]
        dist = TransformedDistribution(base_dist, transform)
        dist = Independent(dist, 1)  # Introduces dependence between actions dimension
        dist = SampleDist(
            dist
        )  # because after transform a distribution, some methods may become invalid, such as entropy, mean and mode, we need SmapleDist to approximate it.

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        if with_logprob:
            log_prob = base_dist.log_prob(action).sum(-1)
            correction = torch.log(1 - action.pow(2) + 1e-6).sum(-1)
            log_prob = log_prob - correction
            return action, log_prob

        return action, None
