import numpy as np
import torch
from world_models.memory.dreamer_memory import ReplayBuffer
from world_models.memory.planet_memory import Memory, Episode


class TestReplayBuffer:
    def test_initialization(self):
        size = 100
        obs_shape = (3, 64, 64)
        action_size = 2
        seq_len = 10
        batch_size = 4
        buffer = ReplayBuffer(size, obs_shape, action_size, seq_len, batch_size)
        assert buffer.size == size
        assert buffer.obs_shape == obs_shape
        assert buffer.action_size == action_size
        assert buffer.seq_len == seq_len
        assert buffer.batch_size == batch_size
        assert buffer.idx == 0
        assert not buffer.full

    def test_add(self):
        size = 10
        obs_shape = (3, 64, 64)
        action_size = 2
        seq_len = 5
        batch_size = 2
        buffer = ReplayBuffer(size, obs_shape, action_size, seq_len, batch_size)
        obs = {"image": np.random.randint(0, 256, obs_shape, dtype=np.uint8)}
        ac = np.random.randn(action_size).astype(np.float32)
        rew = 1.0
        done = False
        buffer.add(obs, ac, rew, done)
        assert buffer.idx == 1
        assert buffer.steps == 1
        assert buffer.episodes == 0

    def test_sample(self):
        size = 20
        obs_shape = (3, 64, 64)
        action_size = 2
        seq_len = 5
        batch_size = 2
        buffer = ReplayBuffer(size, obs_shape, action_size, seq_len, batch_size)
        # Add some data
        for i in range(10):
            obs = {"image": np.random.randint(0, 256, obs_shape, dtype=np.uint8)}
            ac = np.random.randn(action_size).astype(np.float32)
            rew = float(i)
            done = i == 9
            buffer.add(obs, ac, rew, done)
        obs_batch, acs_batch, rews_batch, terms_batch = buffer.sample()
        assert obs_batch.shape == (seq_len, batch_size, *obs_shape)
        assert acs_batch.shape == (seq_len, batch_size, action_size)
        assert rews_batch.shape == (seq_len, batch_size)
        assert terms_batch.shape == (seq_len, batch_size)


class TestEpisode:
    def test_initialization(self):
        episode = Episode()
        assert episode.size == 0
        assert len(episode.x) == 0

    def test_append(self):
        episode = Episode()
        obs = torch.randn(3, 64, 64)
        act = torch.randn(2)
        reward = 1.0
        terminal = False
        episode.append(obs, act, reward, terminal)
        assert episode.size == 1
        assert len(episode.x) == 1

    def test_terminate(self):
        episode = Episode()
        obs = torch.randn(3, 64, 64)
        act = torch.randn(2)
        reward = 1.0
        terminal = False
        episode.append(obs, act, reward, terminal)
        final_obs = torch.randn(3, 64, 64)
        episode.terminate(final_obs)
        assert isinstance(episode.x, np.ndarray)
        assert episode.x.shape[0] == 2  # initial + final


class TestMemory:
    def test_initialization(self):
        memory = Memory(size=10)
        assert len(memory) == 0

    def test_append_episode(self):
        memory = Memory(size=10)
        episode = Episode()
        obs = torch.randn(3, 64, 64)
        act = torch.randn(2)
        reward = 1.0
        terminal = True
        episode.append(obs, act, reward, terminal)
        final_obs = torch.randn(3, 64, 64)
        episode.terminate(final_obs)
        memory.append([episode])
        assert len(memory) == 1

    def test_sample(self):
        memory = Memory(size=10)
        # Add a dummy episode
        episode = Episode()
        for _ in range(5):
            obs = torch.randn(3, 64, 64)
            act = torch.randn(2)
            reward = 1.0
            terminal = False
            episode.append(obs, act, reward, terminal)
        final_obs = torch.randn(3, 64, 64)
        episode.terminate(final_obs)
        memory.append([episode])
        batch = memory.sample(batch_size=1, tracelen=3)
        assert len(batch) == 4  # obs, act, rew, term
