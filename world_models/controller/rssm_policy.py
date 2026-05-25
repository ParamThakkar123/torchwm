import torch
from torch.distributions import Normal


class RSSMPolicy:
    """Model-predictive controller using Cross-Entropy Method (CEM) with RSSM.

    Plans actions by optimizing a sequence of future actions in the RSSM's
    latent space. Uses Cross-Entropy Method to refine action sequences based
    on predicted returns.

    Algorithm:
        1. Initialize Gaussian distribution over action sequences
        2. Sample N candidate action sequences
        3. Rollout each sequence in RSSM latent space
        4. Score by predicted cumulative rewards
        5. Keep top K candidates, fit Gaussian to them
        6. Repeat for T iterations
        7. Execute first action from best sequence

    Why latent space planning?
        - Images are high-dimensional; latent states are compact
        - Enables thousands of rollouts in parallel
        - Dynamics model is more accurate in latent space

    Args:
        model: RSSM instance for latent dynamics
        planning_horizon: Number of future steps to plan (H)
        num_candidates: Number of action sequences to sample (N)
        num_iterations: CEM refinement iterations (T)
        top_candidates: Number of best candidates to keep (K)
        device: torch device

    Usage with Planet agent:
        policy = RSSMPolicy(
            model=rssm,
            planning_horizon=12,
            num_candidates=1000,
            num_iterations=8,
            top_candidates=100,
            device='cuda'
        )

        policy.reset()
        action = policy.poll(observation)  # (1, action_dim)

        # For continuous control:
        next_obs, reward, done, info = env.step(action)

    Comparison with Dreamer:
        - RSSMPolicy: Online planning, chooses actions by optimization at each step
        - DreamerActor: Train actor network to predict actions from states
        - Dreamer is more sample-efficient for complex tasks; CEM is more flexible
    """

    def __init__(
        self,
        model,
        planning_horizon,
        num_candidates,
        num_iterations,
        top_candidates,
        device,
    ):
        super().__init__()
        self.rssm = model
        self.N = num_candidates
        self.K = top_candidates
        self.T = num_iterations
        self.H = planning_horizon
        self.d = self.rssm.action_size
        self.device = device
        self.state_size = self.rssm.state_size
        self.latent_size = self.rssm.latent_size

    def reset(self):
        self.h = torch.zeros(1, self.state_size).to(self.device)
        self.s = torch.zeros(1, self.latent_size).to(self.device)
        self.a = torch.zeros(1, self.d).to(self.device)

    def _poll(self, obs):
        self.mu = torch.zeros(self.H, self.d).to(self.device)
        self.stddev = torch.ones(self.H, self.d).to(self.device)
        # observation could be of shape [CHW] but only 1 timestep
        assert len(obs.shape) == 3, "obs should be [CHW]"
        self.h, self.s = self.rssm.get_init_state(
            self.rssm.encoder(obs[None]), self.h, self.s, self.a
        )
        for _ in range(self.T):
            rwds = torch.zeros(self.N).to(self.device)
            actions = Normal(self.mu, self.stddev).sample((self.N,))
            h_t = self.h.clone().expand(self.N, -1)
            s_t = self.s.clone().expand(self.N, -1)
            for a_t in torch.unbind(actions, dim=1):
                h_t = self.rssm.deterministic_state_fwd(h_t, s_t, a_t)
                s_t = self.rssm.state_prior(h_t, sample=True)
                rwds += self.rssm.pred_reward(h_t, s_t).squeeze(-1)
            _, k = torch.topk(rwds, self.K, dim=0, largest=True, sorted=False)
            self.mu = actions[k].mean(dim=0)
            self.stddev = actions[k].std(dim=0, unbiased=False)
        self.a = self.mu[0:1]

    def poll(self, observation, explore=False):
        with torch.no_grad():
            self._poll(observation)
            if explore:
                self.a += torch.randn_like(self.a) * 0.3
            return self.a
