from functools import partial
import os
import torch
import numpy as np

from world_models.utils.utils import to_tensor_obs, preprocess_img
from world_models.models.rssm import RecurrentStateSpaceModel
from world_models.controller.rssm_policy import RSSMPolicy
from world_models.controller.rollout_generator import RolloutGenerator
from world_models.utils.utils import (
    TensorBoardMetrics,
    save_video,
    postprocess_img,
    TorchImageEnvWrapper,
)
from world_models.memory.planet_memory import Memory, Episode
from world_models.training.train_planet import train as planet_train


class Planet:
    """
    High-level Planet wrapper.

    Usage example:
      from world_models.models.planet import Planet
      p = Planet(env='CartPole-v1', bit_depth=5)
      p.train(epochs=50)
    """

    def __init__(
        self,
        env,
        bit_depth=5,
        device=None,
        state_size=200,
        latent_size=30,
        embedding_size=1024,
        memory_size=100,
        policy_cfg=None,
        headless=False,
        max_episode_steps=None,
        action_repeats=1,
        results_dir=None,
    ):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bit_depth = bit_depth

        if isinstance(env, str):
            self.env = TorchImageEnvWrapper(env, bit_depth, None, action_repeats)
        elif hasattr(env, "action_size"):
            self.env = env
        else:
            self.env = self._wrap_raw_env(env, bit_depth, action_repeats)

        self.rssm = RecurrentStateSpaceModel(
            self.env.action_size, state_size, latent_size, embedding_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.rssm.parameters(), lr=1e-4, eps=1e-4)

        policy_cfg = policy_cfg or {}
        self.policy = RSSMPolicy(
            model=self.rssm,
            planning_horizon=policy_cfg.get("planning_horizon", 20),
            num_candidates=policy_cfg.get("num_candidates", 1000),
            num_iterations=policy_cfg.get("num_iterations", 10),
            top_candidates=policy_cfg.get("top_candidates", 100),
            device=self.device,
        )

        env_max_steps = (
            max_episode_steps or getattr(self.env, "max_episode_steps", None) or 100
        )
        self.rollout_gen = RolloutGenerator(
            self.env,
            self.device,
            policy=self.policy,
            episode_gen=lambda: Episode(partial(postprocess_img, depth=self.bit_depth)),
            max_episode_steps=env_max_steps,
        )

        self.memory = Memory(memory_size)
        self.summary = None
        self.results_dir = results_dir or "results/planet"

    def _wrap_raw_env(self, env, bit_depth, action_repeats):
        class SimpleEnvWrapper:
            def __init__(self, env, bit_depth, action_repeats):
                self.env = env
                self.bit_depth = bit_depth
                self.action_repeats = action_repeats

            @property
            def action_size(self):
                if hasattr(self.env.action_space, "n"):
                    return 1
                else:
                    return self.env.action_space.shape[0]

            @property
            def observation_size(self):
                return (3, 64, 64)

            @property
            def max_episode_steps(self):
                if (
                    hasattr(self.env, "_max_episode_steps")
                    and self.env._max_episode_steps is not None
                ):
                    return self.env._max_episode_steps
                if hasattr(self.env, "spec") and hasattr(
                    self.env.spec, "max_episode_steps"
                ):
                    return self.env.spec.max_episode_steps
                return 1000

            def sample_random_action(self):
                return torch.tensor(self.env.action_space.sample(), dtype=torch.float32)

            def reset(self):
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                frame = self._get_frame(obs)
                if frame is None or frame.size == 0:
                    raise RuntimeError("Environment returned invalid frame on reset")
                x = to_tensor_obs(frame)
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(
                        f"Invalid tensor before preprocessing: min={x.min()}, max={x.max()}"
                    )
                    raise ValueError("NaN/Inf in observation tensor")
                preprocess_img(x, self.bit_depth)
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(
                        f"Invalid tensor after preprocessing: min={x.min()}, max={x.max()}"
                    )
                    raise ValueError("NaN/Inf after preprocessing")
                return x

            def step(self, action):
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if hasattr(self.env.action_space, "n"):
                    action = int(np.clip(action, 0, self.env.action_space.n - 1))
                obs, reward, term, trunc, info = self.env.step(action)
                done = term or trunc
                frame = self._get_frame(obs)
                x = to_tensor_obs(frame)
                preprocess_img(x, self.bit_depth)
                return x, reward, done, info

            def _get_frame(self, obs):
                frame = self.env.render()
                if isinstance(frame, tuple):
                    frame = frame[0]
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    return frame
                if isinstance(obs, np.ndarray) and obs.ndim == 1:
                    vals = (obs - obs.min()) / (obs.max() - obs.min() + 1e-8)
                    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
                    for i, v in enumerate(vals[:8]):
                        band = int(255 * v)
                        canvas[:, i * 8 : (i + 1) * 8, :] = band
                    return canvas
                return np.zeros((64, 64, 3), dtype=np.uint8)

            def __getattr__(self, name):
                return getattr(self.env, name)

        return SimpleEnvWrapper(env, bit_depth, action_repeats)

    def warmup(self, n_episodes=1, random_policy=True):
        """Collect n_episodes of rollouts into memory (used as warmup)."""
        eps = self.rollout_gen.rollout_n(n=n_episodes, random_policy=random_policy)
        self.memory.append(eps)

    def train(
        self,
        epochs=100,
        steps_per_epoch=150,
        batch_size=32,
        H=50,
        beta=1.0,
        save_every=25,
        record_grads=False,
        results_dir=None,
    ):
        """
        High-level training loop. Delegates single-step training to the existing `train` function.

        This mirrors the behavior in world_models/training/train_planet.py but wrapped as a class method.
        """
        # allow caller to override results dir for this training run
        if results_dir is not None:
            self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.summary = TensorBoardMetrics(self.results_dir)

        if len(self.memory.episodes) == 0:
            self.warmup(n_episodes=1, random_policy=True)

        for ep in range(epochs):
            metrics = {}
            for _ in range(steps_per_epoch):
                train_metrics = planet_train(
                    self.memory,
                    self.rssm.train(),
                    self.optimizer,
                    self.device,
                    N=batch_size,
                    H=H,
                    beta=beta,
                    grads=record_grads,
                )
                for k, v in train_metrics.items():
                    if k not in metrics:
                        metrics[k] = []
                    if isinstance(v, dict):
                        for ik, iv in v.items():
                            metrics.setdefault(f"{k}/{ik}", []).append(iv)
                    else:
                        metrics[k].append(v)

            compact = {}
            for k, vs in metrics.items():
                try:
                    compact[k] = np.mean(vs)
                except Exception:
                    compact[k] = vs
            self.summary.update(compact)

            self.memory.append(self.rollout_gen.rollout_once(explore=True))
            eval_episode, eval_frames, eval_metrics = self.rollout_gen.rollout_eval()
            self.memory.append(eval_episode)
            save_video(eval_frames, self.results_dir, f"vid_{ep+1}")
            self.summary.update(eval_metrics)

            if (ep + 1) % save_every == 0:
                torch.save(
                    self.rssm.state_dict(),
                    os.path.join(self.results_dir, f"ckpt_{ep+1}.pth"),
                )

        return self.results_dir
