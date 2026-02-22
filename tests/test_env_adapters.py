import numpy as np
from unittest.mock import Mock, patch

import gym
import torch

from world_models.configs.dreamer_config import DreamerConfig
from world_models.envs.gym_env import GymImageEnv
from world_models.models.dreamer import make_env
from world_models.utils.utils import TorchImageEnvWrapper


class _FakeDiscreteEnv:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.last_action = None
        self.spec = type("Spec", (), {"max_episode_steps": 5})()

    def reset(self, seed=None):
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), {}

    def step(self, action):
        self.last_action = action
        obs = np.array([0.2, 0.1, 0.0, -0.1], dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        pass


@patch("world_models.models.dreamer.env_wrapper.TimeLimit")
@patch("world_models.models.dreamer.env_wrapper.NormalizeActions")
@patch("world_models.models.dreamer.env_wrapper.ActionRepeat")
@patch("world_models.models.dreamer.DeepMindControlEnv")
def test_make_env_dmc_backend(
    mock_dmc,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "dmc"
    cfg.env = "walker-walk"
    cfg.image_size = (64, 64)

    env = Mock()
    mock_dmc.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_dmc.assert_called_once_with(cfg.env, cfg.seed, size=cfg.image_size)


@patch("world_models.models.dreamer.env_wrapper.TimeLimit")
@patch("world_models.models.dreamer.env_wrapper.NormalizeActions")
@patch("world_models.models.dreamer.env_wrapper.ActionRepeat")
@patch("world_models.models.dreamer.GymImageEnv")
def test_make_env_gym_backend(
    mock_gym_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "gym"
    cfg.env = "Pendulum-v1"
    cfg.image_size = (64, 64)
    cfg.gym_render_mode = "rgb_array"

    env = Mock()
    mock_gym_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_gym_env.assert_called_once_with(
        cfg.env,
        seed=cfg.seed,
        size=cfg.image_size,
        render_mode=cfg.gym_render_mode,
    )


@patch("world_models.models.dreamer.env_wrapper.TimeLimit")
@patch("world_models.models.dreamer.env_wrapper.NormalizeActions")
@patch("world_models.models.dreamer.env_wrapper.ActionRepeat")
@patch("world_models.models.dreamer.UnityMLAgentsEnv")
def test_make_env_unity_backend(
    mock_unity_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "unity_mlagents"
    cfg.unity_file_name = "fake.exe"
    cfg.unity_behavior_name = "Behavior"

    env = Mock()
    mock_unity_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    assert mock_unity_env.call_count == 1
    call_kwargs = mock_unity_env.call_args.kwargs
    assert call_kwargs["file_name"] == cfg.unity_file_name
    assert call_kwargs["behavior_name"] == cfg.unity_behavior_name


def test_gym_image_env_discrete_action_mapping():
    wrapped = GymImageEnv(_FakeDiscreteEnv(), seed=1, size=(64, 64))
    obs = wrapped.reset()
    assert obs["image"].shape == (3, 64, 64)

    action = np.array([-0.2, 0.7, 0.1], dtype=np.float32)
    _, reward, done, info = wrapped.step(action)

    assert wrapped._env.last_action == 1
    assert reward == 1.0
    assert done is False
    assert info["action"].shape == (3,)
    assert np.array_equal(info["action"], np.array([-1.0, 1.0, -1.0], dtype=np.float32))


class _FakeDictObsEnv:
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.spec = type("Spec", (), {"max_episode_steps": 8})()

    def reset(self):
        return {"image": np.zeros((3, 64, 64), dtype=np.uint8)}

    def step(self, action):
        obs = {"image": np.ones((3, 64, 64), dtype=np.uint8) * 32}
        return obs, 0.25, False, {}

    def render(self, *args, **kwargs):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        pass


def test_torch_image_wrapper_accepts_env_instances_and_dict_obs():
    wrapper = TorchImageEnvWrapper(_FakeDictObsEnv(), bit_depth=5)
    obs = wrapper.reset()
    assert torch.is_tensor(obs)
    assert obs.shape == (3, 64, 64)

    nobs, reward, done, info = wrapper.step(np.array([0.1, -0.2], dtype=np.float32))
    assert torch.is_tensor(nobs)
    assert nobs.shape == (3, 64, 64)
    assert reward == 0.5  # action_repeats defaults to 2
    assert done is False
    assert isinstance(info, dict)
