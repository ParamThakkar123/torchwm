World Model Training
===================

Generate data and train world models using your existing implementations.

Data Generation
--------------

First, generate a training dataset:

.. code-block:: python

    from torchwm.sim.envs.basic_env import BasicEnv
    from torchwm.sim.exporters.hdf5 import HDF5Exporter

    config = {
        "physics": {"timestep": 1/60, "substeps": 1, "num_solver_iterations": 50, "gravity_z": -9.81},
        "generator": {"objects": [{"shape": {"type": "box", "size": [0.5, 0.5, 0.5]}, "position": [0, 0, 1], "mass": 1.0}]},
        "camera": {"width": 64, "height": 64, "fov": 60, "position": [1, 1, 1], "target": [0, 0, 0]},
    }

    env = BasicEnv(config)

    with HDF5Exporter("world_model_data.h5", mode="w") as exp:
        for ep in range(1000):
            obs, _ = env.reset(seed=ep)
            frames = [obs]
            actions = []
            rewards = []

            for t in range(100):
                action = [0.0, 0.0, 0.0]  # random or policy
                obs, reward, _, _ = env.step({"body": 0, "action": action, "mode": "torque"})
                frames.append(obs)
                actions.append(action)
                rewards.append(reward)

            exp.add_episode(frames=frames, actions=actions, rewards=rewards, dones=[])

    env.close()

PyTorch Dataset
--------------

Create a PyTorch DataLoader:

.. code-block:: python

    import h5py
    import torch
    from torch.utils.data import Dataset, DataLoader

    class WorldModelDataset(Dataset):
        def __init__(self, h5_path, seq_len=16):
            self.seq_len = seq_len
            with h5py.File(h5_path, 'r') as f:
                self.episodes = list(f.keys())
        
        def __len__(self):
            return len(self.episodes) * 10
        
        def __getitem__(self, idx):
            ep_idx = idx // 10
            start = idx % 80
            
            with h5py.File("world_model_data.h5", 'r') as f:
                ep = f[self.episodes[ep_idx]]
                frames = ep['frames'][start:start+self.seq_len]
                actions = ep['actions'][start:start+self.seq_len]
            
            # Preprocess
            frames = torch.FloatTensor(frames) / 255.0  # [T, H, W, C]
            frames = frames.permute(0, 3, 1, 2)       # [T, C, H, W]
            actions = torch.FloatTensor(actions)
            
            return frames, actions

    dataset = WorldModelDataset("world_model_data.h5")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

Training Loop
-------------

Train your world model:

.. code-block:: python

    for epoch in range(100):
        for batch in loader:
            frames, actions = batch  # [B, T, C, H, W], [B, T, A]
            
            # Encode observations
            z = encoder(frames)  # latent representation
            
            # World model: predict next latent
            z_pred = world_model(z[:, :-1], actions[:, :-1])
            
            # Loss: reconstruction + KL
            recon_loss = ((z_pred.mean - z[:, 1:]) ** 2).mean()
            kl_loss = kl_divergence(z_pred, z[:, 1:])
            loss = recon_loss + 0.1 * kl_loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

With Observation Wrappers
-------------------------

Use built-in wrappers:

.. code-block:: python

    from torchwm.sim.wrappers.observation import FrameStackWrapper, NormalizeWrapper, ToTensorWrapper

    env = BasicEnv(config)
    env = FrameStackWrapper(env, num_stack=4)     # Stack 4 frames
    env = NormalizeWrapper(env, range_min=-1, range_max=1)  # Normalize
    env = ToTensorWrapper(env, device="cuda")      # Convert to torch tensor

    obs, _ = env.reset(seed=0)
    # obs is now a stacked, normalized torch tensor
