import numpy as np


class ReplayBuffer:
    """Fixed-size replay buffer for Dreamer with image observations and transitions.

    Stores (observation, action, reward, terminal) tuples in a ring buffer and
    supports sampling contiguous sequences for world-model training.

    Key Features:
        - Ring buffer with fixed capacity (FIFO eviction when full)
        - Stores raw uint8 images to save memory
        - Samples sequences (not single transitions) for temporal modeling
        - Validates sampled sequences don't span episode boundaries

    Memory Layout:
        - observations: (capacity, C, H, W) uint8 images
        - actions: (capacity, action_dim) float32
        - rewards: (capacity,) float32
        - terminals: (capacity,) float32 (1.0 = terminal, 0.0 = continue)

    Sampling Process:
        1. Random start index (avoiding episode boundaries)
        2. Collect sequence of length seq_len with wraparound
        3. Validate no terminal in middle of sequence
        4. Return batch of sequences

    Usage with Dreamer:
        buffer = ReplayBuffer(
            size=100000,           # Max transitions to store
            obs_shape=(3, 64, 64), # RGB images
            action_size=6,         # Continuous action dim
            seq_len=50,            # Sequence length for training
            batch_size=50          # Parallel sequences per batch
        )

        # Add transitions during interaction
        buffer.add(obs, action, reward, done)

        # Sample batch for world model training
        obs_batch, action_batch, reward_batch, term_batch = buffer.sample()
        # Shapes: (seq_len, batch, C, H, W), (seq_len, batch, action_dim), etc.

    Memory Efficiency:
        - Uses uint8 for images (1 byte per pixel vs 4 for float32)
        - Sequences share observations (overlapping windows)
        - Configurable capacity based on available system memory

    Note:
        The buffer stores observations as {"image": ...} dicts but returns
        just the image arrays for training efficiency.
    """

    def __init__(self, size, obs_shape, action_size, seq_len, batch_size):
        self.size = size
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.idx = 0
        self.full = False
        self.observations = np.empty((size, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.terminals = np.empty((size,), dtype=np.float32)
        self.steps, self.episodes = 0, 0

    def add(self, obs, ac, rew, done):
        """Add a transition to the buffer.

        Args:
            obs: Observation dict with 'image' key containing the observation
            ac: Action taken, shape (action_size,)
            rew: Reward received, scalar
            done: Terminal flag, 1.0 if episode ended, 0.0 otherwise
        """
        self.observations[self.idx] = obs["image"]
        self.actions[self.idx] = ac
        self.rewards[self.idx] = rew
        self.terminals[self.idx] = done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        self.episodes = self.episodes + (1 if done else 0)

    def _sample_idx(self, L):
        """Sample valid starting indices for a sequence of length L.

        Ensures the sampled sequence doesn't span episode boundaries
        by checking that no terminal flags appear in the sequence.

        Args:
            L: Sequence length to validate

        Returns:
            Array of L indices into the buffer
        """
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = self.idx not in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        """Retrieve batch of sequences given indices.

        Args:
            idxs: Starting indices for n sequences, shape (n, L)
            n: Number of sequences (batch size)
            L: Sequence length

        Returns:
            observations: (L, n, C, H, W)
            actions: (L, n, action_dim)
            rewards: (L, n)
            terminals: (L, n)
        """
        vec_idxs = idxs.transpose().reshape(-1)
        observations = self.observations[vec_idxs]
        return (
            observations.reshape(L, n, *observations.shape[1:]),
            self.actions[vec_idxs].reshape(L, n, -1),
            self.rewards[vec_idxs].reshape(L, n),
            self.terminals[vec_idxs].reshape(L, n),
        )

    def sample(self):
        """Sample a batch of sequences for training.

        Returns:
            tuple: (observations, actions, rewards, terminals)
                - observations: (seq_len, batch, C, H, W)
                - actions: (seq_len, batch, action_dim)
                - rewards: (seq_len, batch)
                - terminals: (seq_len, batch)
        """
        n = self.batch_size
        L = self.seq_len
        obs, acs, rews, terms = self._retrieve_batch(
            np.asarray([self._sample_idx(L) for _ in range(n)]), n, L
        )
        return obs, acs, rews, terms


class Memory:
    """Simple deque-based memory for storing transitions.

    Used by PlaNet for online planning. Stores recent transitions and
    provides random sampling for policy updates.

    Args:
        capacity: Maximum number of transitions to store

    Usage:
        memory = Memory(capacity=10000)
        memory.append(obs, action, reward, done, info)
        batch = random.sample(memory, batch_size=32)
    """

    from collections import deque

    def __init__(self, capacity: int = 10000):
        self.memory = self.deque(maxlen=capacity)

    def append(self, *args):
        """Append a transition to memory."""
        self.memory.append(args)

    def sample(self, batch_size: int):
        """Sample random batch from memory."""
        from random import sample

        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Episode:
    """Stores a single episode for PlaNet's imagination and planning.

    An episode is a sequence of (observation, action, reward) tuples
    collected during environment interaction. Episodes are used for
    computing returns and training value functions.

    Args:
        obs: Initial observation
        action: First action (optional)
        reward: Initial reward (optional)
        info: Additional info dict (optional)

    Usage:
        episode = Episode(obs, info=info)
        episode.append(action, obs, reward, done, info)
        episodes = [episode for _ in range(num_episodes)]

        # Use with Planet agent for planning
        imag_state, imag_reward, imag_action = planet.imagine(episodes)
    """

    from collections import namedtuple

    _fields = ["observation", "action", "reward", "terminal", "info"]

    def __init__(self, observation, action=None, reward=None, terminal=None, info=None):
        self.observation = observation
        if action is not None:
            self.action = [action]
        else:
            self.action = []
        if reward is not None:
            self.reward = [reward]
        else:
            self.reward = []
        if terminal is not None:
            self.terminal = [terminal]
        else:
            self.terminal = []
        self.info = info if info is not None else {}

    def append(self, action, observation, reward, terminal, info=None):
        self.action.append(action)
        self.observation.append(observation)
        self.reward.append(reward)
        self.terminal.append(terminal)
        if info is not None:
            for k, v in info.items():
                if k not in self.info:
                    self.info[k] = []
                self.info[k].append(v)

    def __len__(self):
        return len(self.observation)
