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


class _FakeMjModel:
    nu = 2
    nq = 1
    nv = 1

    def __init__(self):
        self.actuator_ctrlrange = np.array([[-2.0, 2.0], [-0.5, 0.5]], dtype=np.float32)
        self.actuator_ctrllimited = np.array([True, True])

    @staticmethod
    def from_xml_string(xml, assets=None):
        assert "<mujoco" in xml
        return _FakeMjModel()

    @staticmethod
    def from_xml_path(path):
        return _FakeMjModel()

    @staticmethod
    def from_binary_path(path):
        return _FakeMjModel()


class _FakeMjData:
    def __init__(self, model):
        self.ctrl = np.zeros((model.nu,), dtype=np.float32)
        self.qpos = np.zeros((model.nq,), dtype=np.float64)
        self.qvel = np.zeros((model.nv,), dtype=np.float64)
        self.time = 0.0


class _FakeRenderer:
    def __init__(self, model, height, width):
        self.height = height
        self.width = width
        self.closed = False

    def update_scene(self, data, camera=None):
        self.camera = camera

    def render(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8) + 7

    def close(self):
        self.closed = True


def test_native_mujoco_image_env_with_mocked_bindings(monkeypatch):
    fake_mujoco = type("FakeMujoco", (), {})()
    fake_mujoco.MjModel = _FakeMjModel
    fake_mujoco.MjData = _FakeMjData
    fake_mujoco.Renderer = _FakeRenderer
    fake_mujoco.mj_resetData = lambda model, data: None
    fake_mujoco.mj_forward = lambda model, data: None

    def fake_step(model, data, nstep=1):
        data.time += 0.01 * nstep

    fake_mujoco.mj_step = fake_step
    monkeypatch.setitem(__import__("sys").modules, "mujoco", fake_mujoco)

    from world_models.envs.mujoco_env import MuJoCoImageEnv

    env = MuJoCoImageEnv(
        xml_string="<mujoco/>",
        size=(32, 40),
        camera="track",
        frame_skip=3,
        reward_fn=lambda model, data, action, info: float(action.sum()),
        terminal_fn=lambda model, data, info: data.time > 0.02,
    )
    obs = env.reset(seed=123)
    assert obs["image"].shape == (3, 32, 40)
    assert env.action_space.shape == (2,)
    obs, reward, done, info = env.step(np.array([3.0, 0.25], dtype=np.float32))
    assert obs["image"].shape == (3, 32, 40)
    assert reward == 2.25
    assert done is True
    assert np.array_equal(info["action"], np.array([2.0, 0.25], dtype=np.float32))


def test_list_gymnasium_robotics_envs_uses_registered_package_envs(monkeypatch):
    import sys
    from importlib.machinery import ModuleSpec
    from types import ModuleType, SimpleNamespace

    import world_models.envs.robotics_env as robotics_env

    fake_robotics = ModuleType("gymnasium_robotics")
    fake_robotics.__spec__ = ModuleSpec("gymnasium_robotics", loader=None)
    monkeypatch.setitem(sys.modules, "gymnasium_robotics", fake_robotics)
    monkeypatch.setattr(
        robotics_env.gym, "register_envs", lambda module: None, raising=False
    )
    monkeypatch.setattr(
        robotics_env.gym.envs,
        "registry",
        {
            "FetchReachDense-v4": SimpleNamespace(
                entry_point="gymnasium_robotics.envs.fetch.reach:MujocoFetchReachEnv"
            ),
            "AntMaze_UMazeDense-v5": SimpleNamespace(
                entry_point="gymnasium_robotics.envs.maze.ant_maze_v5:AntMazeEnv"
            ),
            "CartPole-v1": SimpleNamespace(
                entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv"
            ),
        },
    )

    assert robotics_env.list_gymnasium_robotics_envs() == [
        "AntMaze_UMazeDense-v5",
        "FetchReachDense-v4",
    ]


def test_environment_catalog_exposes_robotics_to_online_world_models(monkeypatch):
    import world_models.catalog as catalog

    monkeypatch.setattr(
        catalog, "_list_available_robotics_envs", lambda: ["FetchReachDense-v4"]
    )
    monkeypatch.setattr(catalog, "_list_available_atari_envs", lambda: ["ALE/Pong-v5"])

    envs_by_model = catalog._build_env_catalog()

    for model_name in (
        "dreamer",
        "dreamerv1",
        "dreamerv2",
        "planet",
        "rssm",
        "iris",
        "diamond",
        "genie",
        "dit",
    ):
        assert "FetchReachDense-v4" in envs_by_model[model_name]

    assert "jepa" not in envs_by_model
    assert "ijepa" not in envs_by_model


@patch("world_models.envs.mujoco_env.GymImageEnv")
@patch("world_models.envs.robotics_env.gym.make")
def test_make_mujoco_env_supports_generic_gymnasium_mujoco_task(
    mock_gym_make,
    mock_gym_image_env,
):
    from world_models.envs.mujoco_env import make_mujoco_env

    base_env = Mock()
    wrapped_env = Mock()
    mock_gym_make.return_value = base_env
    mock_gym_image_env.return_value = wrapped_env

    out_env = make_mujoco_env(
        "Humanoid-v4",
        seed=7,
        size=(32, 32),
        render_mode="rgb_array",
        forward_reward_weight=1.5,
    )

    assert out_env is wrapped_env
    mock_gym_make.assert_called_once_with(
        "Humanoid-v4",
        render_mode="rgb_array",
        forward_reward_weight=1.5,
    )
    mock_gym_image_env.assert_called_once_with(
        base_env,
        seed=7,
        size=(32, 32),
        render_mode="rgb_array",
    )


def test_make_mujoco_env_from_config_builds_xml_string_env(monkeypatch):
    fake_mujoco = type("FakeMujoco", (), {})()
    fake_mujoco.MjModel = _FakeMjModel
    fake_mujoco.MjData = _FakeMjData
    fake_mujoco.Renderer = _FakeRenderer
    fake_mujoco.mj_resetData = lambda model, data: None
    fake_mujoco.mj_forward = lambda model, data: None
    fake_mujoco.mj_step = lambda model, data, nstep=1: None
    monkeypatch.setitem(__import__("sys").modules, "mujoco", fake_mujoco)

    from world_models.envs.mujoco_env import make_mujoco_env_from_config

    cfg = DreamerConfig()
    cfg.mujoco_xml_string = "<mujoco/>"
    cfg.mujoco_camera = "track"
    cfg.mujoco_frame_skip = 2
    env = make_mujoco_env_from_config(cfg, size=(16, 24))

    assert env.observation_space["image"].shape == (3, 16, 24)
    assert env._camera == "track"
    assert env._frame_skip == 2


class _FakeJaxRandom:
    @staticmethod
    def PRNGKey(seed):
        return np.array([seed], dtype=np.int64)

    @staticmethod
    def split(key):
        base = int(np.asarray(key).reshape(-1)[0])
        return np.array([base + 1], dtype=np.int64), np.array(
            [base + 2], dtype=np.int64
        )


class _FakeJax:
    random = _FakeJaxRandom()

    @staticmethod
    def jit(fn):
        return fn

    @staticmethod
    def device_get(value):
        return value


class _FakeBraxState:
    def __init__(self, obs, reward=0.0, done=0.0, metrics=None, info=None):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.reward = np.asarray(reward, dtype=np.float32)
        self.done = np.asarray(done, dtype=np.float32)
        self.metrics = metrics or {}
        self.info = info or {}


class _FakeBraxEnv:
    action_size = 2
    episode_length = 7

    def __init__(self):
        self.last_action = None

    def reset(self, rng):
        return _FakeBraxState(np.array([0.0, 0.5, 1.0], dtype=np.float32))

    def step(self, state, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return _FakeBraxState(
            np.array([1.0, 0.0, -1.0], dtype=np.float32),
            reward=1.25,
            done=0.0,
            metrics={"metric": np.asarray(3.0, dtype=np.float32)},
        )


def test_brax_image_env_adapts_functional_brax_api(monkeypatch):
    from world_models.envs.brax_env import BraxImageEnv

    monkeypatch.setattr(
        "world_models.envs.brax_env._require_module",
        lambda module_name, install_hint, **kwargs: {
            "jax": _FakeJax,
            "jax.numpy": np,
            "brax.envs": Mock(get_environment=lambda *args, **kwargs: _FakeBraxEnv()),
        }[module_name],
    )

    wrapped = BraxImageEnv(
        "ant", seed=0, size=(32, 32), backend="generalized", jit=True
    )
    obs = wrapped.reset()
    assert obs["image"].shape == (3, 32, 32)
    assert wrapped.action_space.shape == (2,)
    assert wrapped.max_episode_steps == 7

    next_obs, reward, done, info = wrapped.step(np.array([2.0, -2.0], dtype=np.float32))

    assert next_obs["image"].shape == (3, 32, 32)
    assert reward == 1.25
    assert done is False
    assert np.array_equal(info["action"], np.array([1.0, -1.0], dtype=np.float32))
    assert info["vector_observation"].shape == (3,)
    assert "discount" in info


@patch("world_models.models.dreamer.env_wrapper.TimeLimit")
@patch("world_models.models.dreamer.env_wrapper.NormalizeActions")
@patch("world_models.models.dreamer.env_wrapper.ActionRepeat")
@patch("world_models.models.dreamer.make_mujoco_env_from_config")
def test_make_env_native_mujoco_backend(
    mock_make_mujoco_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "mujoco"
    cfg.env = "model.xml"
    cfg.image_size = (64, 64)
    cfg.mujoco_camera = "fixed"
    cfg.mujoco_frame_skip = 4
    cfg.mujoco_reset_noise_scale = 0.01

    env = Mock()
    mock_make_mujoco_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_make_mujoco_env.assert_called_once_with(cfg, cfg.image_size)


@patch("world_models.models.dreamer.env_wrapper.TimeLimit")
@patch("world_models.models.dreamer.env_wrapper.NormalizeActions")
@patch("world_models.models.dreamer.env_wrapper.ActionRepeat")
@patch("world_models.models.dreamer.BraxImageEnv")
def test_make_env_brax_backend(
    mock_brax_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "brax"
    cfg.env = "ant"
    cfg.image_size = (64, 64)
    cfg.brax_backend = "mjx"
    cfg.brax_jit = False
    cfg.brax_auto_reset = False
    cfg.time_limit = 100

    env = Mock()
    mock_brax_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_brax_env.assert_called_once_with(
        cfg.env,
        seed=cfg.seed,
        size=cfg.image_size,
        backend=cfg.brax_backend,
        episode_length=cfg.time_limit,
        auto_reset=cfg.brax_auto_reset,
        jit=cfg.brax_jit,
        suppress_warp_warnings=cfg.brax_suppress_warp_warnings,
    )


def test_require_module_filters_warp_messages(monkeypatch, capsys):
    import importlib

    from world_models.envs import brax_env as be

    # Make find_spec always report modules exist.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True)

    def fake_import(name):
        # Simulate noisy import-time prints from MuJoCo/MJX shim.
        print("Some other message")
        print("Failed to import warp: No module named 'warp'")
        print("Failed to import mujoco_warp: No module named 'mujoco_warp'")

        class M:
            pass

        return M

    monkeypatch.setattr(importlib, "import_module", fake_import)

    # When suppression is enabled, only the non-warp line should be replayed.
    mod = be._require_module("brax.envs", "hint", suppress_warp_warnings=True)
    captured = capsys.readouterr()
    assert "Some other message" in captured.out
    assert "Failed to import warp:" not in captured.out
    assert "Failed to import mujoco_warp:" not in captured.out


@patch("world_models.envs.robotics_env.GymImageEnv")
@patch("world_models.envs.robotics_env.gym.make")
def test_make_robotics_env_registers_gymnasium_robotics(
    mock_gym_make,
    mock_gym_image_env,
    monkeypatch,
):
    import sys
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    import world_models.envs.robotics_env as robotics_env
    from world_models.envs.robotics_env import make_robotics_env

    fake_robotics = ModuleType("gymnasium_robotics")
    fake_robotics.__spec__ = ModuleSpec("gymnasium_robotics", loader=None)
    monkeypatch.setitem(sys.modules, "gymnasium_robotics", fake_robotics)
    monkeypatch.setattr(
        robotics_env.gym, "register_envs", lambda module: None, raising=False
    )
    base_env = Mock()
    wrapped_env = Mock()
    mock_gym_make.return_value = base_env
    mock_gym_image_env.return_value = wrapped_env

    out_env = make_robotics_env(
        "HalfCheetah-v2",
        seed=3,
        size=(32, 32),
        render_mode="rgb_array",
        reset_noise_scale=0.2,
    )

    assert out_env is wrapped_env
    mock_gym_make.assert_called_once_with(
        "HalfCheetah-v2",
        render_mode="rgb_array",
        reset_noise_scale=0.2,
    )
    mock_gym_image_env.assert_called_once_with(
        base_env,
        seed=3,
        size=(32, 32),
        render_mode="rgb_array",
    )


@patch("world_models.envs.mujoco_env.GymImageEnv")
@patch("world_models.envs.robotics_env.gym.make")
def test_make_mujoco_env_falls_back_to_gymnasium_robotics_for_legacy_ids(
    mock_gym_make,
    mock_gym_image_env,
    monkeypatch,
):
    import sys
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    import world_models.envs.robotics_env as robotics_env
    from world_models.envs.mujoco_env import make_mujoco_env

    fake_robotics = ModuleType("gymnasium_robotics")
    fake_robotics.__spec__ = ModuleSpec("gymnasium_robotics", loader=None)
    monkeypatch.setitem(sys.modules, "gymnasium_robotics", fake_robotics)
    monkeypatch.setattr(
        robotics_env.gym, "register_envs", lambda module: None, raising=False
    )
    base_env = Mock()
    wrapped_env = Mock()
    mock_gym_make.side_effect = [
        ImportError(
            "The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project."
        ),
        base_env,
    ]
    mock_gym_image_env.return_value = wrapped_env

    out_env = make_mujoco_env("HalfCheetah-v2", seed=5, size=(16, 16))

    assert out_env is wrapped_env
    assert mock_gym_make.call_count == 2
    mock_gym_make.assert_called_with("HalfCheetah-v2", render_mode="rgb_array")
    mock_gym_image_env.assert_called_once_with(
        base_env,
        seed=5,
        size=(16, 16),
        render_mode="rgb_array",
    )


@patch("world_models.models.dreamer.env_wrapper.TimeLimit")
@patch("world_models.models.dreamer.env_wrapper.NormalizeActions")
@patch("world_models.models.dreamer.env_wrapper.ActionRepeat")
@patch("world_models.models.dreamer.make_robotics_env")
def test_make_env_robotics_backend(
    mock_robotics_env,
    mock_repeat,
    mock_normalize,
    mock_time_limit,
):
    cfg = DreamerConfig()
    cfg.env_backend = "robotics"
    cfg.env = "HalfCheetah-v2"
    cfg.image_size = (64, 64)
    cfg.gym_render_mode = "rgb_array"

    env = Mock()
    mock_robotics_env.return_value = env
    mock_repeat.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_normalize.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env
    mock_time_limit.side_effect = lambda wrapped_env, *args, **kwargs: wrapped_env

    out_env = make_env(cfg)

    assert out_env is env
    mock_robotics_env.assert_called_once_with(
        "HalfCheetah-v2",
        seed=cfg.seed,
        size=cfg.image_size,
        render_mode="rgb_array",
    )


def test_catalog_queries_gymnasium_registry_at_runtime(monkeypatch):
    import sys
    from types import SimpleNamespace

    import world_models.catalog as catalog

    fake_gym = SimpleNamespace(
        envs=SimpleNamespace(
            registry={
                "CartPole-v1": SimpleNamespace(
                    id="CartPole-v1",
                    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
                    namespace=None,
                ),
                "ExampleControl-v9": SimpleNamespace(
                    id="ExampleControl-v9",
                    entry_point="example_package:ExampleEnv",
                    namespace=None,
                ),
                "ALE/Pong-v5": SimpleNamespace(
                    id="ALE/Pong-v5",
                    entry_point="ale_py.env:PongEnv",
                    namespace="ALE",
                ),
                "FetchReachDense-v4": SimpleNamespace(
                    id="FetchReachDense-v4",
                    entry_point="gymnasium_robotics.envs.fetch.reach:MujocoFetchReachEnv",
                    namespace=None,
                ),
            }
        )
    )
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)

    assert catalog._list_available_gymnasium_envs() == [
        "CartPole-v1",
        "ExampleControl-v9",
    ]


def test_environment_catalog_uses_runtime_gymnasium_envs(monkeypatch):
    import world_models.catalog as catalog

    monkeypatch.setattr(
        catalog, "_list_available_gymnasium_envs", lambda: ["DynamicEnv-v9"]
    )
    monkeypatch.setattr(
        catalog, "_list_available_robotics_envs", lambda: ["FetchReach-v4"]
    )
    monkeypatch.setattr(catalog, "_list_available_atari_envs", lambda: ["ALE/Pong-v5"])

    envs_by_model = catalog._build_env_catalog()

    assert "DynamicEnv-v9" in envs_by_model["dreamer"]
    assert "DynamicEnv-v9" in envs_by_model["planet"]
    assert "DynamicEnv-v9" not in envs_by_model["iris"]
    assert "ALE/Pong-v5" in envs_by_model["iris"]
    assert "FetchReach-v4" in envs_by_model["diamond"]
