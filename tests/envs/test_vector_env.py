import numpy as np
import pytest
import torch
from gym import spaces

from world_models.envs.vector_env import SimWorker, TorchVectorizedEnv, WorkerError
from world_models.training.rl_harness import PPOTrainer


class CountingImageEnv:
    """Small deterministic image env used by vectorized-env tests."""

    def __init__(self, episode_length=2):
        self.episode_length = episode_length
        self.observation_space = {
            "image": spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8)
        }
        self.action_space = spaces.Discrete(4)
        self.seed_value = 0
        self.step_count = 0
        self.closed = False

    def seed(self, seed):
        self.seed_value = int(seed)

    def _obs(self):
        value = (self.seed_value + self.step_count) % 256
        return {"image": np.full((3, 64, 64), value, dtype=np.uint8)}

    def reset(self):
        self.step_count = 0
        return self._obs()

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.episode_length
        info = {"seed": self.seed_value, "step": self.step_count, "action": int(action)}
        return self._obs(), float(self.seed_value + int(action)), done, info

    def render(self):
        return np.full((64, 64, 3), self.seed_value % 256, dtype=np.uint8)

    def close(self):
        self.closed = True


def make_counting_env():
    return CountingImageEnv()


class FailingStepEnv(CountingImageEnv):
    """Env that raises once its step method is called."""

    def step(self, action):
        raise RuntimeError("intentional step failure")


def make_failing_step_env():
    return FailingStepEnv()


@pytest.fixture
def vec_env():
    env = TorchVectorizedEnv(
        make_counting_env,
        num_workers=2,
        envs_per_worker=2,
        seed=10,
    )
    try:
        yield env
    finally:
        env.close()


class TestSimWorker:
    def test_reset_and_step_batch_track_each_env_done_independently(self):
        worker = SimWorker(
            worker_id=0,
            env_factory=make_counting_env,
            num_envs=2,
            command_queue=None,
            result_queue=None,
            seed=0,
        )
        worker.envs = [
            CountingImageEnv(episode_length=1),
            CountingImageEnv(episode_length=3),
        ]
        worker.dones = [False, False]
        worker.last_obs = [None, None]

        reset_results = worker._reset_batch()
        assert [result["obs"]["image"][0, 0, 0] for result in reset_results] == [0, 0]
        assert worker.dones == [False, False]

        first_step = worker._step_batch([np.array(1), np.array(2)])
        assert [result["done"] for result in first_step] == [True, False]
        assert [result["reward"] for result in first_step] == [1.0, 2.0]

        second_step = worker._step_batch([np.array(3), np.array(1)])
        assert second_step[0]["obs"] is first_step[0]["obs"]
        assert second_step[0]["reward"] == 0.0
        assert second_step[0]["done"] is True
        assert second_step[0]["info"] == {}
        assert second_step[1]["done"] is False
        assert second_step[1]["info"]["step"] == 2

    def test_run_assigns_zero_seed_to_first_env(self):
        import multiprocessing as mp

        command_queue = mp.Queue()
        result_queue = mp.Queue()
        worker = SimWorker(
            worker_id=0,
            env_factory=make_counting_env,
            num_envs=2,
            command_queue=command_queue,
            result_queue=result_queue,
            seed=0,
        )
        worker.start()
        try:
            command_queue.put(("reset", None))
            cmd, results = result_queue.get(timeout=5.0)
            assert cmd == "reset_result"
            seeded_pixels = [
                int(result["obs"]["image"][0, 0, 0]) for result in results
            ]
            assert seeded_pixels == [0, 1]
        finally:
            command_queue.put(("close", None))
            worker.join(timeout=5.0)
            command_queue.close()
            result_queue.close()
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5.0)


class TestTorchVectorizedEnv:
    def test_reset_batch_returns_normalized_images_in_worker_order(self, vec_env):
        batch = vec_env.reset_batch()

        assert batch["obs"]["image"].shape == (4, 3, 64, 64)
        assert batch["obs"]["image"].dtype == torch.float32
        expected = torch.tensor([10, 11, 12, 13], dtype=torch.float32) / 255.0
        observed = batch["obs"]["image"][:, 0, 0, 0]
        torch.testing.assert_close(observed, expected)

    def test_step_batch_preserves_batch_shapes_rewards_dones_and_infos(self, vec_env):
        vec_env.reset_batch()
        actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        first_step = vec_env.step_batch(actions)
        assert first_step["obs"]["image"].shape == (4, 3, 64, 64)
        assert first_step["reward"].shape == (4,)
        assert first_step["done"].shape == (4,)
        assert len(first_step["info"]) == 4
        torch.testing.assert_close(
            first_step["reward"], torch.tensor([10.0, 12.0, 14.0, 16.0])
        )
        assert first_step["done"].tolist() == [False, False, False, False]
        assert [info["seed"] for info in first_step["info"]] == [10, 11, 12, 13]

        second_step = vec_env.step_batch(actions)
        assert second_step["done"].tolist() == [True, True, True, True]

        third_step = vec_env.step_batch(actions)
        torch.testing.assert_close(third_step["reward"], torch.zeros(4))
        assert third_step["done"].tolist() == [True, True, True, True]
        assert third_step["info"] == [{}, {}, {}, {}]

        reset = vec_env.reset_batch()
        torch.testing.assert_close(
            reset["obs"]["image"][:, 0, 0, 0],
            torch.tensor([10, 11, 12, 13], dtype=torch.float32) / 255.0,
        )

    def test_render_batch_returns_one_frame_per_env(self, vec_env):
        frames = vec_env.render_batch()

        assert len(frames) == 4
        assert all(frame.shape == (64, 64, 3) for frame in frames)
        assert [int(frame[0, 0, 0]) for frame in frames] == [10, 11, 12, 13]

    def test_worker_errors_include_traceback_and_restart_for_recovery(self):
        env = TorchVectorizedEnv(
            make_failing_step_env,
            num_workers=1,
            envs_per_worker=1,
            seed=5,
        )
        try:
            env.reset_batch()

            with pytest.raises(WorkerError) as exc_info:
                env.step_batch(torch.tensor([0], dtype=torch.long))

            message = str(exc_info.value)
            assert "Worker 0 failed during step" in message
            assert "RuntimeError: intentional step failure" in message
            assert "Traceback" in message

            # The failed process is replaced so callers can recover with a reset.
            assert env.workers[0].is_alive()
            reset = env.reset_batch()
            torch.testing.assert_close(
                reset["obs"]["image"][:, 0, 0, 0],
                torch.tensor([5], dtype=torch.float32) / 255.0,
            )
        finally:
            env.close()


class TestPPOTrainerVectorizedHarness:
    def test_collect_trajectories_uses_vectorized_reset_and_step_contract(self, vec_env):
        torch.manual_seed(0)
        trainer = PPOTrainer(
            vec_env,
            device="cpu",
            num_epochs=1,
            batch_size=4,
        )

        trajectories = trainer.collect_trajectories(num_steps=8)

        assert trajectories["obs"].shape == (2, 4, 3, 64, 64)
        assert trajectories["actions"].shape == (2, 4)
        assert trajectories["log_probs"].shape == (2, 4)
        assert trajectories["rewards"].shape == (2, 4)
        assert trajectories["values"].shape == (2, 4)
        assert trajectories["dones"].shape == (2, 4)
        assert trajectories["dones"][0].tolist() == [False, False, False, False]
        assert trajectories["dones"][1].tolist() == [True, True, True, True]

    def test_train_step_updates_policy_parameters(self, vec_env):
        torch.manual_seed(0)
        trainer = PPOTrainer(
            vec_env,
            device="cpu",
            num_epochs=1,
            batch_size=4,
        )
        trajectories = trainer.collect_trajectories(num_steps=8)
        before = [
            parameter.detach().clone() for parameter in trainer.policy.parameters()
        ]

        trainer.train_step(trajectories)

        after = list(trainer.policy.parameters())
        assert any(not torch.equal(old, new) for old, new in zip(before, after))
