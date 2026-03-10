from world_models.controller.rollout_generator import RolloutGenerator
from world_models.controller.rssm_policy import RSSMPolicy


class TestRolloutGenerator:
    def test_initialization(self):
        env = None  # Mock env
        device = "cpu"
        policy = None
        max_episode_steps = 100
        generator = RolloutGenerator(env, device, policy, max_episode_steps)
        assert generator.env == env
        assert generator.device == device
        assert generator.policy == policy
        assert generator.max_episode_steps == max_episode_steps


class TestRSSMPolicy:
    def test_initialization(self):
        model = None  # Mock model
        planning_horizon = 10
        num_candidates = 100
        num_iterations = 5
        top_candidates = 10
        device = "cpu"
        policy = RSSMPolicy(
            model,
            planning_horizon,
            num_candidates,
            num_iterations,
            top_candidates,
            device,
        )
        assert policy.rssm == model
        assert policy.H == planning_horizon
        assert policy.N == num_candidates
        assert policy.T == num_iterations
        assert policy.K == top_candidates
        assert policy.device == device
