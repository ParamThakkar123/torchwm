import pytest
import numpy as np
from unittest.mock import Mock, patch
from world_models.models.dreamer import DreamerAgent
from world_models.models.dreamer_rssm import RSSM
from world_models.configs.dreamer_config import DreamerConfig


class TestDreamerAgent:
    @pytest.fixture
    def config(self):
        config = DreamerConfig()
        config.env = "cartpole_balance"
        config.seed = 42
        config.total_steps = 1000
        config.seed_steps = 1
        config.action_repeat = 1
        config.restore = False
        config.buffer_size = 10
        return config

    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_initialization(self, mock_logger, mock_make_env, config):
        mock_env = Mock()
        mock_obs_space = Mock()
        mock_obs_space.shape = (3, 64, 64)  # Assuming image shape for DMC env
        mock_env.observation_space = {"image": mock_obs_space}
        mock_action_space = Mock()
        mock_action_space.shape = (2,)  # Assuming action shape
        mock_env.action_space = mock_action_space
        mock_make_env.return_value = mock_env

        config.buffer_size = 10  # Reduce for test
        agent = DreamerAgent(config)

        assert agent.args == config
        assert isinstance(agent.dreamer.rssm, RSSM)
        assert agent.train_env == mock_env
        assert agent.test_env == mock_env
        assert agent.logger == mock_logger.return_value

    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_evaluate(self, mock_logger, mock_make_env, config):
        mock_env = Mock()
        mock_obs_space = Mock()
        mock_obs_space.shape = (3, 64, 64)
        mock_env.observation_space = {"image": mock_obs_space}
        mock_action_space = Mock()
        mock_action_space.shape = (2,)
        mock_env.action_space = mock_action_space
        mock_make_env.return_value = mock_env

        config.buffer_size = 10  # Reduce for test
        agent = DreamerAgent(config)
        agent.dreamer.evaluate = Mock(
            return_value=(np.array([4.0, 5.0]), np.array([[]]), None)
        )

        agent.evaluate()

        agent.dreamer.evaluate.assert_called_once_with(
            agent.test_env, config.test_episodes, render=True
        )
        mock_logger.return_value.dump_scalars_to_pickle.assert_called_once()
        mock_logger.return_value.log_videos.assert_called_once()

    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_initialization_with_invalid_arg(self, mock_logger, mock_make_env, config):
        mock_env = Mock()
        mock_obs_space = Mock()
        mock_obs_space.shape = (3, 64, 64)
        mock_env.observation_space = {"image": mock_obs_space}
        mock_action_space = Mock()
        mock_action_space.shape = (2,)
        mock_env.action_space = mock_action_space
        mock_make_env.return_value = mock_env

        with pytest.raises(ValueError, match="Invalid argument: invalid_arg"):
            DreamerAgent(config, invalid_arg="test")

    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_export_uses_dreamer_actor_default(
        self, mock_logger, mock_make_env, config, monkeypatch, tmp_path
    ):
        import world_models.export as export_utils

        mock_env = Mock()
        mock_obs_space = Mock()
        mock_obs_space.shape = (3, 64, 64)
        mock_env.observation_space = {"image": mock_obs_space}
        mock_action_space = Mock()
        mock_action_space.shape = (2,)
        mock_env.action_space = mock_action_space
        mock_make_env.return_value = mock_env

        calls = {}

        def fake_export_model(module, path, **kwargs):
            calls["module"] = module
            calls["path"] = path
            calls["kwargs"] = kwargs
            return path

        monkeypatch.setattr(export_utils, "export_model", fake_export_model)
        agent = DreamerAgent(config)
        out_path = tmp_path / "dreamer.onnx"

        returned = agent.export(out_path, format="onnx")

        assert returned == out_path
        assert calls["kwargs"]["format"] == "onnx"
        assert calls["kwargs"]["example_inputs"].shape == (
            1,
            config.stoch_size + config.deter_size,
        )
        assert calls["module"](calls["kwargs"]["example_inputs"]).shape == (1, 2)
