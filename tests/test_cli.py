from click.testing import CliRunner
from tools import cli


def test_version_shows_package_version():
    runner = CliRunner()
    res = runner.invoke(cli.app, ["version"])
    assert res.exit_code == 0
    assert "0.4.0" in res.output


def test_envs_list_outputs_backends():
    runner = CliRunner()
    res = runner.invoke(cli.app, ["envs", "list"])
    assert res.exit_code == 0
    # ui.server declares these backends; ensure at least 'gym' appears
    assert "gym:" in res.output


def test_datasets_list_empty_dir(tmp_path):
    runner = CliRunner()
    # point to an empty temporary folder
    res = runner.invoke(cli.app, ["datasets", "list", str(tmp_path)])
    assert res.exit_code == 0
    assert "No datasets found" in res.output


def test_datasets_list_with_subdir(tmp_path):
    d = tmp_path / "sample"
    d.mkdir()
    runner = CliRunner()
    res = runner.invoke(cli.app, ["datasets", "list", str(tmp_path)])
    assert res.exit_code == 0
    assert "- sample" in res.output


def test_train_unknown_model_errors():
    runner = CliRunner()
    res = runner.invoke(cli.app, ["train", "notamodel"])
    assert res.exit_code == 1
    assert "Unknown model" in res.output


def test_benchmark_requires_agent_or_all_agents():
    runner = CliRunner()
    res = runner.invoke(cli.app, ["benchmark", "--game", "ALE/Pong-v5"])
    assert res.exit_code == 1
    assert "Either --agent or --all-agents" in res.output


def test_benchmark_single_agent_requires_checkpoint():
    runner = CliRunner()
    res = runner.invoke(
        cli.app, ["benchmark", "--agent", "iris", "--game", "ALE/Pong-v5"]
    )
    assert res.exit_code == 1
    assert "--checkpoint is required" in res.output


def test_benchmark_single_agent_runs_with_checkpoint(monkeypatch, tmp_path):
    calls = {}

    class FakeBenchmarkRunner:
        def __init__(self, adapter_cls, out_dir):
            calls["adapter_cls"] = adapter_cls
            calls["out_dir"] = out_dir

        def run(self, **kwargs):
            calls["run_kwargs"] = kwargs
            return {"aggregate": {}}

    class FakeMultiAgentBenchmarkRunner:
        pass

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(
        cli,
        "_load_benchmark_runtime",
        lambda: (
            {"iris": object},
            FakeBenchmarkRunner,
            FakeMultiAgentBenchmarkRunner,
            FakeTorch,
        ),
    )

    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "benchmark",
            "--agent",
            "iris",
            "--game",
            "ALE/Pong-v5",
            "--checkpoint",
            str(tmp_path / "iris.pt"),
            "--seeds",
            "0,2",
            "--episodes",
            "3",
            "--out-dir",
            str(tmp_path / "bench"),
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code == 0
    assert "Benchmark finished" in res.output
    assert calls["out_dir"] == str(tmp_path / "bench")
    assert calls["run_kwargs"]["env_spec"] == {"game": "ALE/Pong-v5"}
    assert calls["run_kwargs"]["seeds"] == [0, 2]
    assert calls["run_kwargs"]["num_episodes"] == 3
    assert calls["run_kwargs"]["checkpoint"] == str(tmp_path / "iris.pt")
    assert calls["run_kwargs"]["extra_kwargs"]["device"] == "cpu"


def test_console_entrypoint_run_is_exported():
    assert callable(cli.run)


def test_replay_browse_summary_flat_hdf5(tmp_path):
    import pytest

    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")

    path = tmp_path / "buffer.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("observations", data=np.zeros((4, 8, 8, 3), dtype=np.uint8))
        f.create_dataset("actions", data=np.array([0, 1, 2, 3], dtype=np.int64))
        f.create_dataset("rewards", data=np.array([1.0, 2.0, -1.0, 4.0], dtype=np.float32))
        f.create_dataset("dones", data=np.array([False, True, False, True]))

    runner = CliRunner()
    res = runner.invoke(cli.app, ["replay", "browse", str(path), "--summary"])

    assert res.exit_code == 0
    assert "Replay buffer:" in res.output
    assert "2 episodes" in res.output
    assert "4 steps" in res.output
    assert "episode_0 length=2 return=3.0" in res.output
    assert "episode_1 length=2 return=3.0" in res.output


def test_replay_browse_can_render_frame_png(tmp_path):
    import pytest

    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL.Image")
    from tools.replay_browser import render_frame_png

    path = tmp_path / "buffer.h5"
    with h5py.File(path, "w") as f:
        episode = f.create_group("episode_000")
        episode.create_dataset("obs", data=np.full((1, 3, 8, 8), 255, dtype=np.uint8))
        episode.create_dataset("actions", data=np.array([7], dtype=np.int64))
        episode.create_dataset("rewards", data=np.array([1.5], dtype=np.float32))
        episode.create_dataset("dones", data=np.array([True]))

    png = render_frame_png(path, 0, 0)

    assert png.startswith(b"\x89PNG")
