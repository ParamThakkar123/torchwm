import os
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
