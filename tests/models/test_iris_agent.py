import pytest
import torch

from world_models.models.iris_agent import IRISAgent, compute_lambda_return
from world_models.configs.iris_config import IRISConfig


class TestComputeLambdaReturn:
    def test_compute_lambda_return_shapes(self):
        B, T = 4, 10
        rewards = torch.randn(B, T)
        values = torch.randn(B, T + 1)
        discounts = torch.full((B, T), 0.99)
        lambda_coef = 0.95

        result = compute_lambda_return(rewards, values, discounts, lambda_coef)

        assert result.shape == (B, T)

    def test_compute_lambda_return_single_batch(self):
        B, T = 1, 5
        rewards = torch.randn(B, T)
        values = torch.randn(B, T + 1)
        discounts = torch.full((B, T), 0.99)
        lambda_coef = 0.95

        result = compute_lambda_return(rewards, values, discounts, lambda_coef)

        assert result.shape == (B, T)

    def test_compute_lambda_return_matches_expected(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        values = torch.tensor([[10.0, 8.0, 6.0, 4.0], [5.0, 4.0, 3.0, 2.0]])
        discounts = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        lambda_coef = 0.0

        result = compute_lambda_return(rewards, values, discounts, lambda_coef)

        expected_last = rewards[:, -1] + discounts[:, -1] * values[:, -1]
        assert torch.allclose(result[:, -1], expected_last)


class TestIRISAgentImagineRollout:
    @pytest.fixture
    def config(self):
        config = IRISConfig()
        config.vocab_size = 32
        config.tokens_per_frame = 16
        config.token_embedding_dim = 128
        config.frame_channels = 3
        config.encoder_channels = 32
        config.decoder_channels = 32
        config.frame_shape = (3, 64, 64)
        config.transformer_layers = 2
        config.transformer_heads = 4
        config.transformer_embed_dim = 128
        config.discount = 0.99
        return config

    @pytest.fixture
    def agent(self, config):
        device = torch.device("cpu")
        agent = IRISAgent(config, action_size=4, device=device)
        return agent

    def test_imagine_rollout_reward_shape(self, agent):
        B, C, H, W = 2, 3, 64, 64
        initial_frame = torch.randn(B, C, H, W)
        horizon = 5

        with torch.no_grad():
            trajectory = agent.imagine_rollout(initial_frame, horizon=horizon)

        assert trajectory["rewards"].shape == (B, horizon)

    def test_imagine_rollout_frames_shape(self, agent):
        B, C, H, W = 2, 3, 64, 64
        initial_frame = torch.randn(B, C, H, W)
        horizon = 5

        with torch.no_grad():
            trajectory = agent.imagine_rollout(initial_frame, horizon=horizon)

        assert trajectory["frames"].shape == (B, horizon + 1, C, H, W)

    def test_imagine_rollout_actions_shape(self, agent):
        B, C, H, W = 2, 3, 64, 64
        initial_frame = torch.randn(B, C, H, W)
        horizon = 5

        with torch.no_grad():
            trajectory = agent.imagine_rollout(initial_frame, horizon=horizon)

        assert trajectory["actions"].shape == (B, horizon)

    def test_update_actor_critic_with_imagined_trajectory(self, agent):
        B, T, C, H, W = 2, 5, 3, 64, 64
        frames = torch.randn(B, T + 1, C, H, W)
        actions = torch.randint(0, 4, (B, T))
        rewards = torch.randn(B, T)

        imagined = {
            "frames": frames,
            "actions": actions,
            "rewards": rewards,
        }

        metrics = agent.update_actor_critic(imagined)

        assert "actor_loss" in metrics
        assert "value_loss" in metrics
        assert "total_loss" in metrics
