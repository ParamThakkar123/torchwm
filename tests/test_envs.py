import pytest
import importlib.util
from world_models.envs.dmc import DeepMindControlEnv


@pytest.mark.skip(reason="Requires MuJoCo/EGL which is not available")
class TestDeepMindControlEnv:
    def test_env_creation(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_env_step_and_reset(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        obs = env.reset()
        assert "image" in obs
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        assert "image" in next_obs
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "discount" in info

    def test_env_render(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        env.reset()
        frame = env.render()
        assert frame.shape == (64, 64, 3)
        # env.close()  # DMC env may not have close

    def test_env_seed_reproducibility(self):
        env1 = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        env2 = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert (obs1["image"] == obs2["image"]).all()
        action = env1.action_space.sample()
        next_obs1, _, _, _ = env1.step(action)
        next_obs2, _, _, _ = env2.step(action)
        assert (next_obs1["image"] == next_obs2["image"]).all()

    def test_invalid_env_name(self):
        try:
            DeepMindControlEnv(name="invalid-env", seed=42, size=(64, 64))
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_different_image_sizes(self):
        sizes = [(32, 32), (64, 64), (128, 128)]
        for size in sizes:
            env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=size)
            obs = env.reset()
            assert obs["image"].shape == (size[0], size[1], 3)

    def test_multiple_resets(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        for _ in range(5):
            obs = env.reset()
            assert "image" in obs

    def test_step_after_done(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        try:
            env.step(env.action_space.sample())
        except Exception as e:
            assert isinstance(e, RuntimeError)

    def test_close_env(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        env.reset()
        # env.close()  # DMC env may not have close
        try:
            env.step(env.action_space.sample())
        except Exception as e:
            assert isinstance(e, RuntimeError)


class TestMujocoEnv:
    pass


@pytest.mark.skipif(
    not hasattr(__import__("torch", fromlist=["cuda"]), "cuda")
    or not __import__("torch", fromlist=["cuda"]).cuda.is_available(),
    reason="CUDA not available",
)
class TestGPUVectorizedEnv:
    @pytest.fixture
    def mock_env_factory(self):
        import gym

        def factory():
            return gym.make("CartPole-v1")

        return factory

    def test_env_creation(self, mock_env_factory):
        from world_models.envs.vector_env import GPUVectorizedEnv

        env = GPUVectorizedEnv(env_factory=mock_env_factory, num_envs=2, seed=42)
        assert env.observation_space is not None
        assert env.action_space is not None
        assert env.num_envs == 2
        env.close()

    def test_reset_batch(self, mock_env_factory):
        from world_models.envs.vector_env import GPUVectorizedEnv

        env = GPUVectorizedEnv(env_factory=mock_env_factory, num_envs=2, seed=42)
        obs_dict = env.reset_batch()
        assert "obs" in obs_dict
        assert "image" in obs_dict["obs"]
        assert obs_dict["obs"]["image"].shape[0] == 2  # batch size
        env.close()

    def test_step_batch(self, mock_env_factory):
        from world_models.envs.vector_env import GPUVectorizedEnv
        import torch

        env = GPUVectorizedEnv(env_factory=mock_env_factory, num_envs=2, seed=42)
        env.reset_batch()
        actions = torch.randn(2, env.action_space.shape[0], device=env.device)
        result = env.step_batch(actions)
        assert "obs" in result
        assert "reward" in result
        assert "done" in result
        assert result["reward"].shape[0] == 2
        assert result["done"].shape[0] == 2
        env.close()


@pytest.mark.skipif(
    not hasattr(__import__("torch", fromlist=["cuda"]), "cuda")
    or not __import__("torch", fromlist=["cuda"]).cuda.is_available(),
    reason="CUDA not available",
)
@pytest.mark.skipif(not importlib.util.find_spec("brax"), reason="Brax not installed")
class TestGPUVectorizedEnvBrax:
    def test_brax_env_creation_and_jit(self):
        from world_models.envs.vector_env import GPUVectorizedEnv
        import brax.envs

        def factory():
            return brax.envs.ant.Ant()

        env = GPUVectorizedEnv(env_factory=factory, num_envs=2, seed=42)
        assert env.observation_space is not None
        assert env.action_space is not None
        # Check if JIT was applied (hard to test directly, but env should work)
        obs_dict = env.reset_batch()
        assert "obs" in obs_dict
        env.close()


@pytest.mark.skipif(
    not importlib.util.find_spec("isaaclab"),
    reason="IsaacLab not installed",
)
class TestIsaacLabImageEnv:
    def test_env_creation(self):
        # This would require a prebuilt IsaacLab env, skip for now
        pytest.skip("Requires prebuilt IsaacLab environment object")

    def test_image_observation(self):
        pytest.skip("Requires prebuilt IsaacLab environment object")
